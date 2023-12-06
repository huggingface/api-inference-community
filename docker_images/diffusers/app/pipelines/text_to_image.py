import json
import logging
import os
from typing import TYPE_CHECKING
import importlib

import torch
from app import idle, lora, timing, validation
from app.pipelines import Pipeline
from diffusers import (
    AutoencoderKL,
    AutoPipelineForText2Image,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
)
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from huggingface_hub import file_download, hf_api, hf_hub_download, model_info, utils


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from PIL import Image


class TextToImagePipeline(Pipeline, lora.LoRAPipelineMixin):
    def __init__(self, model_id: str):
        self.current_lora_adapter = None
        self.model_id = None
        self.use_auth_token = os.getenv("HF_API_TOKEN")
        # This should allow us to make the image work with private models when no token is provided, if the said model
        # is already in local cache
        self.offline_preferred = validation.str_to_bool(os.getenv("OFFLINE_PREFERRED"))
        fetched = False
        if self.offline_preferred:
            cache_root = os.getenv(
                "DIFFUSERS_CACHE", os.getenv("HUGGINGFACE_HUB_CACHE", "")
            )
            folder_name = file_download.repo_folder_name(
                repo_id=model_id, repo_type="model"
            )
            folder_path = os.path.join(cache_root, folder_name)
            logger.debug("Cache folder path %s", folder_path)
            filename = os.path.join(folder_path, "hub_model_info.json")
            try:
                with open(filename, "r") as f:
                    model_data = json.load(f)
            except OSError:
                logger.info(
                    "No cached model info found in file %s found for model %s. Fetching on the hub",
                    filename,
                    model_id,
                )
            else:
                model_data = hf_api.ModelInfo(**model_data)
                fetched = True

        if not fetched:
            model_data = model_info(model_id, token=self.use_auth_token)

        kwargs = (
            {"safety_checker": None}
            if model_id.startswith("hf-internal-testing/")
            else {}
        )
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16

        has_model_index = any(
            file.rfilename == "model_index.json" for file in model_data.siblings
        )

        if self._is_lora(model_data):
            model_type = "LoraModel"
        elif has_model_index:
            fetched = False
            if self.offline_preferred:
                try:
                    config_file = hf_hub_download(
                        model_id,
                        "model_index.json",
                        token=self.use_auth_token,
                        local_files_only=True,
                    )
                except utils.LocalEntryNotFoundError:
                    logger.info("Unable to fetch model index in local cache")
                else:
                    fetched = True

            if not fetched:
                config_file = hf_hub_download(
                    model_id,
                    "model_index.json",
                    token=self.use_auth_token,
                )
            with open(config_file, "r") as f:
                config_dict = json.load(f)
            model_type = config_dict.get("_class_name", None)
        else:
            raise ValueError("Model type not found")

        if model_type == "LoraModel":
            model_to_load = model_data.cardData["base_model"]
            self.model_id = model_to_load
            if not model_to_load:
                raise ValueError(
                    "No `base_model` found. Please include a `base_model` on your README.md tags"
                )

            weight_name = self._get_lora_weight_name(model_data)
            self._load_sd_with_sdxl_fix(model_to_load, **kwargs)
            self.ldm.load_lora_weights(
                model_id, weight_name=weight_name, use_auth_token=self.use_auth_token
            )
            self.current_lora_adapter = model_id
            self._fuse_or_raise()
            logger.info("LoRA adapter %s loaded", model_id)
        else:
            if model_id == "stabilityai/stable-diffusion-xl-base-1.0":
                self._load_sd_with_sdxl_fix(model_id, **kwargs)
            else:
                self.ldm = AutoPipelineForText2Image.from_pretrained(
                    model_id, use_auth_token=self.use_auth_token, **kwargs
                )
            self.model_id = model_id

        self.is_karras_compatible = (
            self.ldm.__class__.__init__.__annotations__.get("scheduler", None)
            == KarrasDiffusionSchedulers
        )
        if self.is_karras_compatible:
            self.ldm.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.ldm.scheduler.config
            )

        if not idle.UNLOAD_IDLE:
            self._model_to_gpu()

    def _load_sd_with_sdxl_fix(self, model_id, **kwargs):
        if model_id == "stabilityai/stable-diffusion-xl-base-1.0":
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch.float16,  # load fp16 fix VAE
            )
            kwargs["vae"] = vae
            kwargs["variant"] = "fp16"

        self.ldm = DiffusionPipeline.from_pretrained(
            model_id, use_auth_token=self.use_auth_token, **kwargs
        )

    @timing.timing
    def _model_to_gpu(self):
        if torch.cuda.is_available():
            self.ldm.to("cuda")

    def __call__(self, inputs: str, **kwargs) -> "Image.Image":
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`PIL.Image.Image` with the raw image representation as PIL.
        """

        #Check if users set a custom scheduler and pop if from the kwargs if so
        custom_scheduler = None
        if "scheduler" in kwargs:
            custom_scheduler = kwargs["scheduler"]
            kwargs.pop("scheduler")
        
        if custom_scheduler:
            compatibles = self.ldm.compatibles
            #Check if the scheduler is compatible
            is_compatible_scheduler = [cls for cls in compatibles if cls.__name__ == custom_scheduler]
            #In case of a compatible scheduler, swap to that for inference
            if(is_compatible_scheduler):
                #Import the scheduler dynamically
                SchedulerClass = getattr(importlib.import_module("diffusers.schedulers"), custom_scheduler)
                self.ldm.scheduler = SchedulerClass.from_config(self.ldm.scheduler.config)
                
        self._load_lora_adapter(kwargs)

        if idle.UNLOAD_IDLE:
            with idle.request_witnesses():
                self._model_to_gpu()
                resp = self._process_req(inputs, **kwargs)
        else:
            resp = self._process_req(inputs, **kwargs)
        return resp

    def _process_req(self, inputs, **kwargs):
        # only one image per prompt is supported
        kwargs["num_images_per_prompt"] = 1

        if self.is_karras_compatible and "num_inference_steps" not in kwargs:
            kwargs["num_inference_steps"] = 20

        images = self.ldm(inputs, **kwargs)["images"]
        return images[0]
