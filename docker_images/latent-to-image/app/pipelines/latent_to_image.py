import logging
import os
from typing import TYPE_CHECKING

import torch
from app import idle, offline, timing, validation
from app.pipelines import Pipeline
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from PIL import Image


class LatentToImagePipeline(Pipeline, offline.OfflineBestEffortMixin):
    def __init__(self, model_id: str):
        self.model_id = None
        self.current_tokens_loaded = 0
        self.use_auth_token = os.getenv("HF_API_TOKEN")
        # This should allow us to make the image work with private models when no token is provided, if the said model
        # is already in local cache
        self.offline_preferred = validation.str_to_bool(os.getenv("OFFLINE_PREFERRED"))
        model_data = self._hub_model_info(model_id)

        kwargs = {}
        env_dtype = os.getenv("TORCH_DTYPE", "float32")
        if env_dtype:
            kwargs["torch_dtype"] = getattr(torch, env_dtype)
        elif torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16

        has_model_index = any(
            file.rfilename == "model_index.json" for file in model_data.siblings
        )

        if has_model_index:
            kwargs["subfolder"] = "vae"

        self.vae = AutoencoderKL.from_pretrained(model_id, **kwargs).eval()
        self.dtype = kwargs["torch_dtype"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        if not idle.UNLOAD_IDLE:
            self._model_to_gpu()

    @timing.timing
    def _model_to_gpu(self):
        if torch.cuda.is_available():
            self.vae.to("cuda")

    def __call__(self, inputs: torch.Tensor, **kwargs) -> "Image.Image":
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

        latents = inputs.to(self.device, self.dtype)

        with torch.no_grad():
            image = self.vae.decode(latents, return_dict=False)[0]

        image = self.image_processor.postprocess(image, output_type="pil")

        return image[0]
