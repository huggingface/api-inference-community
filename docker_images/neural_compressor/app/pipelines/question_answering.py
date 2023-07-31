from typing import Any, Dict, List

from app.pipelines import Pipeline
from optimum.intel import INCModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import (
    QuestionAnsweringPipeline as TransformersQuestionAnsweringPipeline,
)


class QuestionAnsweringPipeline(Pipeline):
    _MODEL_CLASS = INCModelForQuestionAnswering
    _PIPELINE_CLASS = TransformersQuestionAnsweringPipeline

    def __init__(self, model_id: str):
        super().__init__(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.pipeline = self._PIPELINE_CLASS(model=self.model, tokenizer=tokenizer)

    def __call__(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """
        Args:
            inputs (:obj:`dict`):
                a dictionary containing two keys, 'question' being the question being asked and 'context' being some text containing the answer.
        Return:
            A :obj:`dict`:. The object return should be like {"answer": "XXX", "start": 3, "end": 6, "score": 0.82} containing :
                - "answer": the extracted answer from the `context`.
                - "start": the offset within `context` leading to `answer`. context[start:stop] == answer
                - "end": the ending offset within `context` leading to `answer`. context[start:stop] === answer
                - "score": A score between 0 and 1 describing how confident the model is for this answer.
        """
        return self.pipeline(**inputs)
