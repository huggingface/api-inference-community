import numpy as np
import torch
from app.pipelines import Pipeline
from diffusers import DiffusionPipeline
from PIL import Image


class TextToImagePipeline(Pipeline):
    def __init__(self, model_id: str):
        self.pipeline = DiffusionPipeline.from_pretrained(model_id)
        self.generator = torch.manual_seed(42)

    def __call__(self, inputs: str) -> "Image.Image":
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`PIL.Image` with the raw image representation as PIL.
        """
        kwargs = {"eta": 0.3, "guidance_scale": 6.0, "num_inference_steps": 5}
        image = self.pipeline(
            [inputs], generator=self.generator, torch_device="cpu", **kwargs
        )

        image_processed = image.cpu().permute(0, 2, 3, 1)
        image_processed = image_processed * 255.0
        image_processed = image_processed.numpy().astype(np.uint8)
        return Image.fromarray(image_processed[0])
