from typing import Dict, List

from hezar.models import Model

from app.pipelines import Pipeline


class TextToTextPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model = Model.load(model_id)

    def __call__(self, inputs: str) -> List[Dict[str, str]]:
        """
        Args:
            inputs (:obj:`str`):
                The input text
        Return:
            A :obj:`list`:. The list contains a single item that is a dict {"text": the model output}
        """
        model_outputs = self.model.predict(inputs)
        outputs = [o.dict() for o in model_outputs]
        return outputs
