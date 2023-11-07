from typing import TYPE_CHECKING, Any, Dict, List

from hezar import Model

from app.pipelines import Pipeline


if TYPE_CHECKING:
    from PIL import Image


class ImageToTextPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model = Model.load(model_id)

    def __call__(self, inputs: "Image.Image") -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`PIL.Image`):
                The raw image representation as PIL.
                No transformation made whatsoever from the input. Make all necessary transformations here.
        Return:
            A :obj:`list`:. The list contains a single item that is a dict {"text": the model output}
        """
        model_outputs = self.model.predict(inputs)
        return model_outputs
