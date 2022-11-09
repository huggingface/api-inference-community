import os
from typing import TYPE_CHECKING

import torch
from app.pipelines import Pipeline
from diffusers import DiffusionPipeline


if TYPE_CHECKING:
    from PIL import Image


class TextToImagePipeline(Pipeline):
    def __init__(self, model_id: str):

        kwargs = (
            {"safety_checker": None}
            if model_id.startswith("hf-internal-testing/")
            else {}
        )

        self.ldm = DiffusionPipeline.from_pretrained(
            model_id,
            use_auth_token=os.getenv("HF_API_TOKEN"),
            torch_dtype=torch.float16,
            **kwargs
        )
        if torch.cuda.is_available():
            self.ldm.to("cuda")

    def __call__(self, inputs: str) -> "Image.Image":
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`PIL.Image` with the raw image representation as PIL.
        """
        images = self.ldm([inputs])["images"]
        return images[0]
