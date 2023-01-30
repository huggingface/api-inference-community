import os
from typing import TYPE_CHECKING

import torch
from app.pipelines import Pipeline
from diffusers import (
    AltDiffusionPipeline,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)
from huggingface_hub import model_info


if TYPE_CHECKING:
    from PIL import Image


class TextToImagePipeline(Pipeline):
    def __init__(self, model_id: str):

        model_data = model_info(model_id, token=os.getenv("HF_API_TOKEN"))
        is_lora = any(
            file.rfilename == "pytorch_lora_weights.bin" for file in model_data.siblings
        )

        model_to_load = model_data.cardData["base_model"] if is_lora else model_id

        kwargs = (
            {"safety_checker": None}
            if model_id.startswith("hf-internal-testing/")
            else {}
        )

        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16

        self.ldm = DiffusionPipeline.from_pretrained(
            model_to_load, use_auth_token=os.getenv("HF_API_TOKEN"), **kwargs
        )
        if torch.cuda.is_available():
            self.ldm.to("cuda")
            self.ldm.enable_xformers_memory_efficient_attention()
            self.ldm.unet.to(memory_format=torch.channels_last)

        if is_lora:
            self.ldm.unet.load_attn_procs(
                model_id, use_auth_token=os.getenv("HF_API_TOKEN")
            )

        if isinstance(self.ldm, (StableDiffusionPipeline, AltDiffusionPipeline)):
            self.ldm.scheduler = DPMSolverMultistepScheduler.from_config(
                self.ldm.scheduler.config
            )

    def __call__(self, inputs: str, inference_steps=25) -> "Image.Image":
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`PIL.Image` with the raw image representation as PIL.
        """
        if isinstance(self.ldm, (StableDiffusionPipeline, AltDiffusionPipeline)):
            images = self.ldm([inputs], num_inference_steps=inference_steps)["images"]
        else:
            images = self.ldm([inputs])["images"]
        return images[0]
