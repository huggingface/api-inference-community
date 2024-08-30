import importlib
import json
import logging
import os
from typing import TYPE_CHECKING

import torch
from app import idle, lora, offline, timing, validation
from app.pipelines import Pipeline
from diffusers import (
    AutoencoderKL,
    AutoPipelineForText2Image,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
)
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from PIL import Image


class TextToImagePipeline(
    Pipeline, lora.LoRAPipelineMixin, offline.OfflineBestEffortMixin
):
    def __init__(self, model_id: str):
        self.current_lora_adapter = None
        self.model_id = None
        self.current_tokens_loaded = 0
        self.use_auth_token = os.getenv("HF_API_TOKEN")
        # This should allow us to make the image work with private models when no token is provided, if the said model
        # is already in local cache
        self.offline_preferred = validation.str_to_bool(os.getenv("OFFLINE_PREFERRED"))
        model_data = self._hub_model_info(model_id)

        kwargs = (
            {"safety_checker": None}
            if model_id.startswith("hf-internal-testing/")
            else {}
        )
        env_dtype = os.getenv("TORCH_DTYPE")
        if env_dtype:
            kwargs["torch_dtype"] = getattr(torch, env_dtype)
        elif torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16

        has_model_index = any(
            file.rfilename == "model_index.json" for file in model_data.siblings
        )

        if self._is_lora(model_data):
            model_type = "LoraModel"
        elif has_model_index:
            config_file = self._hub_repo_file(model_id, "model_index.json")
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
            self._load_sd_with_sdxl_fix(model_to_load, **kwargs)
            # The lora will actually be lazily loaded on the fly per request
            self.current_lora_adapter = None
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

        self.default_scheduler = self.ldm.scheduler

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

        # Check if users set a custom scheduler and pop if from the kwargs if so
        custom_scheduler = None
        if "scheduler" in kwargs:
            custom_scheduler = kwargs["scheduler"]
            kwargs.pop("scheduler")

        if custom_scheduler:
            compatibles = self.ldm.scheduler.compatibles
            # Check if the scheduler is compatible
            is_compatible_scheduler = [
                cls for cls in compatibles if cls.__name__ == custom_scheduler
            ]
            # In case of a compatible scheduler, swap to that for inference
            if is_compatible_scheduler:
                # Import the scheduler dynamically
                SchedulerClass = getattr(
                    importlib.import_module("diffusers.schedulers"), custom_scheduler
                )
                self.ldm.scheduler = SchedulerClass.from_config(
                    self.ldm.scheduler.config
                )
            else:
                logger.info("%s scheduler not loaded: incompatible", custom_scheduler)
                self.ldm.scheduler = self.default_scheduler
        else:
            self.ldm.scheduler = self.default_scheduler

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

        if "num_inference_steps" not in kwargs:
            default_num_steps = os.getenv("DEFAULT_NUM_INFERENCE_STEPS")
            if default_num_steps:
                kwargs["num_inference_steps"] = int(default_num_steps)
            elif self.is_karras_compatible:
                kwargs["num_inference_steps"] = 20
            # Else, don't specify anything, leave the default behaviour

        if "seed" in kwargs:
            seed = int(kwargs["seed"])
            generator = torch.Generator().manual_seed(seed)
            kwargs["generator"] = generator
            kwargs.pop("seed")

        images = self.ldm(inputs, **kwargs)["images"]
        return images[0]
