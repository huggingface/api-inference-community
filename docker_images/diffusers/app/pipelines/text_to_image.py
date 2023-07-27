import json
import logging
import os
from typing import TYPE_CHECKING

import torch
from app import idle, timing
from app.pipelines import Pipeline
from diffusers import (
    AltDiffusionPipeline,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    KandinskyPipeline,
    KandinskyPriorPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
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

        elif model_type == "KandinskyPipeline":
            model_to_load = "kandinsky-community/kandinsky-2-1-prior"
            self.ldm = KandinskyPipeline.from_pretrained(
                model_id, use_auth_token=use_auth_token, **kwargs
            )
            self.prior = KandinskyPriorPipeline.from_pretrained(
                model_to_load, use_auth_token=use_auth_token, **kwargs
            )
        else:
            self.ldm = DiffusionPipeline.from_pretrained(
                model_id, use_auth_token=use_auth_token, **kwargs
            )

        if isinstance(
            self.ldm,
            (StableDiffusionXLPipeline, StableDiffusionPipeline, AltDiffusionPipeline),
        ):
            self.ldm.scheduler = DPMSolverMultistepScheduler.from_config(
                self.ldm.scheduler.config
            )

        if not idle.UNLOAD_IDLE:
            self._model_to_gpu()

    @timing.timing
    def _model_to_gpu(self):
        if torch.cuda.is_available():
            self.ldm.to("cuda")
            if isinstance(self.ldm, (KandinskyPipeline)):
                self.prior.to("cuda")

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
        if isinstance(
            self.ldm,
            (StableDiffusionXLPipeline, StableDiffusionPipeline, AltDiffusionPipeline),
        ):
            if "num_inference_steps" not in kwargs:
                kwargs["num_inference_steps"] = 25
            images = self.ldm(inputs, **kwargs)["images"]
            return images[0]
        elif isinstance(self.ldm, (KandinskyPipeline)):
            if "num_inference_steps" not in kwargs:
                kwargs["num_inference_steps"] = 50
            # not all args are supported by the prior
            prior_args = {
                "num_inference_steps": kwargs["num_inference_steps"],
                "num_images_per_prompt": kwargs["num_images_per_prompt"],
                "negative_prompt": kwargs.get("negative_prompt", None),
                "guidance_scale": kwargs.get("guidance_scale", 4),
            }
            image_emb, zero_image_emb = self.prior(inputs, **prior_args).to_tuple()
            images = self.ldm(
                inputs,
                image_embeds=image_emb,
                negative_image_embeds=zero_image_emb,
                **kwargs,
            )["images"]
            return images[0]
        else:
            raise ValueError("Model type not found or pipeline not implemented")
