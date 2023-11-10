from typing import Dict, List

from hezar.models import Model

from app.pipelines import Pipeline


class TextClassificationPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        self.model = Model.load(model_id)

    def __call__(self, inputs: str) -> List[List[Dict[str, float]]]:
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`dict`:. The object returned should be a list of one list like [[{"label": 0.9939950108528137}]] containing:
                - "label": A string representing what the label/class is. There can be multiple labels.
                - "score": A score between 0 and 1 describing how confident the model is for this label/class.
        """
        outputs = self.model.predict(inputs, top_k=len(self.model.config.id2label))
        outputs = [[o.dict() for o in output] for output in outputs]
        return outputs
