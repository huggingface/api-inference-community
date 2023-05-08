from typing import Dict, List

from app.pipelines import Pipeline
from optimum.intel import INCModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import (
    TextClassificationPipeline as TransformersTextClassificationPipeline,
)


class TextClassificationPipeline(Pipeline):
    _MODEL_CLASS = INCModelForSequenceClassification
    _PIPELINE_CLASS = TransformersTextClassificationPipeline

    def __init__(self, model_id: str):
        super().__init__(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.pipeline = self._PIPELINE_CLASS(model=self.model, tokenizer=tokenizer)

    def __call__(self, inputs: str) -> List[Dict[str, float]]:
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`list`:. The object returned should be like [{"label": 0.9939950108528137}] containing :
                - "label": A string representing what the label/class is. There can be multiple labels.
                - "score": A score between 0 and 1 describing how confident the model is for this label/class.
        """
        return self.pipeline(inputs, return_all_scores=True)