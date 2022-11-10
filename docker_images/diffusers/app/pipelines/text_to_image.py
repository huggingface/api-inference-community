import os
from typing import TYPE_CHECKING

import torch
from app.pipelines import Pipeline
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, StableDiffusionPipeline


if TYPE_CHECKING:
    from PIL import Image


class TextToImagePipeline(Pipeline):
    def __init__(self, model_id: str):

        kwargs = (
            {"safety_checker": None}
            if model_id.startswith("hf-internal-testing/")
            else {}
        )

        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16

        self.ldm = DiffusionPipeline.from_pretrained(
            model_id, use_auth_token=os.getenv("HF_API_TOKEN"), **kwargs
        )
        if torch.cuda.is_available():
            self.ldm.to("cuda")

        if isinstance(self.ldm, StableDiffusionPipeline):
            scheduler = EulerDiscreteScheduler.from_pretrained(
                model_id,
                subfolder="scheduler",
                use_auth_token=os.getenv("HF_API_TOKEN"),
            )
            self.ldm.scheduler = scheduler
            self.ldm.enable_xformers_memory_efficient_attention()

    def __call__(self, inputs: str, inference_steps=25) -> "Image.Image":
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`PIL.Image` with the raw image representation as PIL.
        """
        if isinstance(self.ldm, StableDiffusionPipeline):
            images = self.ldm([inputs], num_inference_steps=inference_steps)["images"]
        else:
            images = self.ldm([inputs])["images"]
        return images[0]
