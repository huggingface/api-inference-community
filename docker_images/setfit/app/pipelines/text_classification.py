from typing import Dict, List

from app.pipelines import Pipeline
from setfit import SetFitModel


class TextClassificationPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ) -> None:
        self.model = SetFitModel.from_pretrained(model_id)

    def __call__(self, inputs: str) -> List[Dict[str, float]]:
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`list`:. The object returned should be a list of one list like [[{"label": 0.9939950108528137}]] containing:
                - "label": A string representing what the label/class is. There can be multiple labels.
                - "score": A score between 0 and 1 describing how confident the model is for this label/class.
        """
        probs = self.model.predict_proba([inputs], as_numpy=True)
        if probs.ndim == 2:
            id2label = {}
            if hasattr(self.model, "id2label"):
                id2label = self.model.id2label
            return [
                [
                    {"label": id2label.get(idx, idx), "score": prob}
                    for idx, prob in enumerate(probs[0])
                ]
            ]
