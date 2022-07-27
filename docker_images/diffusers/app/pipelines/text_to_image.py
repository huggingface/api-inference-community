from typing import TYPE_CHECKING

from app.pipelines import Pipeline
from diffusers import DiffusionPipeline


if TYPE_CHECKING:
    from PIL import Image


class TextToImagePipeline(Pipeline):
    def __init__(self, model_id: str):
        self.ldm = DiffusionPipeline.from_pretrained(model_id)

    def __call__(self, inputs: str) -> "Image.Image":
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`PIL.Image` with the raw image representation as PIL.
        """
        images = self.ldm([inputs], num_inference_steps=50, eta=0.3, guidance_scale=6)[
            "sample"
        ]
        return images[0]
