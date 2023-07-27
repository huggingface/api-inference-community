import json
import logging
import os
from typing import TYPE_CHECKING

import torch
from app import idle, timing
from app.pipelines import Pipeline
from diffusers import (
    AutoPipelineForText2Image,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from huggingface_hub import hf_hub_download, model_info


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from PIL import Image


class TextToImagePipeline(Pipeline):
    def __init__(self, model_id: str):
        use_auth_token = os.getenv("HF_API_TOKEN")
        model_data = model_info(model_id, token=use_auth_token)

        kwargs = (
            {"safety_checker": None}
            if model_id.startswith("hf-internal-testing/")
            else {}
        )
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16

        model_type = None
        is_lora = any(
            file.rfilename == "pytorch_lora_weights.bin" for file in model_data.siblings
        )
        has_model_index = any(
            file.rfilename == "model_index.json" for file in model_data.siblings
        )

        if is_lora:
            model_type = "LoraModel"
        elif has_model_index:
            config_file = hf_hub_download(
                model_id, "model_index.json", token=use_auth_token
            )
            with open(config_file, "r") as f:
                config_dict = json.load(f)
            model_type = config_dict.get("_class_name", None)
        else:
            raise ValueError("Model type not found")

        if model_type == "LoraModel":
            model_to_load = model_data.cardData["base_model"]

            self.ldm = DiffusionPipeline.from_pretrained(
                model_to_load, use_auth_token=use_auth_token, **kwargs
            )
            self.ldm.load_lora_weights(model_id, use_auth_token=use_auth_token)
        else:
            self.ldm = AutoPipelineForText2Image.from_pretrained(
                model_id, use_auth_token=use_auth_token, **kwargs
            )

        self.is_karras_compatible = (
            self.ldm.__class__.__init__.__annotations__.get("scheduler", None)
            == KarrasDiffusionSchedulers
        )
        if self.is_karras_compatible:
            self.ldm.scheduler = DPMSolverMultistepScheduler.from_config(
                self.ldm.scheduler.config
            )

        if not idle.UNLOAD_IDLE:
            self._model_to_gpu()

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
            kwargs["num_inference_steps"] = 25

        images = self.ldm(inputs, **kwargs)["images"]
        return images[0]
