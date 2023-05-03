import json
import os

import torch
from app.pipelines import Pipeline
from diffusers import (
    AltDiffusionImg2ImgPipeline,
    ControlNetModel,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
)
from huggingface_hub import hf_hub_download, model_info
from PIL import Image


class ImageToImagePipeline(Pipeline):
    def __init__(self, model_id: str):
        model_data = model_info(model_id, token=os.getenv("HF_API_TOKEN"))

        kwargs = (
            {"safety_checker": None}
            if model_id.startswith("hf-internal-testing/")
            else {}
        )
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16

        has_config = any(
            file.rfilename == "config.json" for file in model_data.siblings
        )

        is_controlnet = False
        if has_config:
            config_file = hf_hub_download(
                model_id, "config.json", token=os.getenv("HF_API_TOKEN")
            )
            with open(config_file, "r") as f:
                config_dict = json.load(f)

            is_controlnet = config_dict.get("_class_name", None) == "ControlNetModel"

        if is_controlnet:
            model_to_load = model_data.cardData["base_model"]
            controlnet = ControlNetModel.from_pretrained(
                model_id, use_auth_token=os.getenv("HF_API_TOKEN"), **kwargs
            )

            self.ldm = StableDiffusionControlNetPipeline.from_pretrained(
                model_to_load,
                controlnet=controlnet,
                use_auth_token=os.getenv("HF_API_TOKEN"),
                **kwargs,
            )
        else:
            self.ldm = DiffusionPipeline.from_pretrained(
                model_id, use_auth_token=os.getenv("HF_API_TOKEN"), **kwargs
            )

        if torch.cuda.is_available():
            self.ldm.to("cuda")
            self.ldm.enable_xformers_memory_efficient_attention()

        if isinstance(
            self.ldm,
            (
                StableDiffusionImg2ImgPipeline,
                AltDiffusionImg2ImgPipeline,
                StableDiffusionControlNetPipeline,
            ),
        ):
            self.ldm.scheduler = DPMSolverMultistepScheduler.from_config(
                self.ldm.scheduler.config
            )

    def __call__(self, image: Image.Image, inputs: str = "", **kwargs) -> "Image.Image":
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
            image (:obj:`PIL.Image.Image`):
                a condition image
        Return:
            A :obj:`PIL.Image.Image` with the raw image representation as PIL.
        """

        if isinstance(
            self.ldm,
            (
                StableDiffusionImg2ImgPipeline,
                AltDiffusionImg2ImgPipeline,
                StableDiffusionControlNetPipeline,
            ),
        ):
            if "num_inference_steps" not in kwargs:
                kwargs["num_inference_steps"] = 25
            images = self.ldm(
                inputs,
                image,
                **kwargs,
            )["images"]
        else:
            images = self.ldm(inputs, image, **kwargs)["images"]
        return images[0]
