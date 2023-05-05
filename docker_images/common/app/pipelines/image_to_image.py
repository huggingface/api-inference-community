from typing import TYPE_CHECKING, Optional

from app.pipelines import Pipeline


if TYPE_CHECKING:
    from PIL import Image


class ImageToImagePipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need for inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        raise NotImplementedError(
            "Please implement ImageToImagePipeline.__init__ function"
        )

    def __call__(self, image: Image.Image, prompt: Optional[str] = "") -> "Image.Image":
        """
        Args:
            image (:obj:`PIL.Image.Image`):
                a condition image
            prompt (:obj:`str`, *optional*):
                a string containing some text
        Return:
            A :obj:`PIL.Image` with the raw image representation as PIL.
        """
        # IMPLEMENT_THIS
        raise NotImplementedError(
            "Please implement ImageToImagePipeline.__call__ function"
        )
