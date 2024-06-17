from typing import Dict, List

from app.pipelines import Pipeline
from transformers import SummarizationPipeline as TransformersSummarizationPipeline


class SummarizationPipeline(Pipeline):
    def __init__(self, adapter_id: str):
        self.pipeline = self._load_pipeline_instance(
            TransformersSummarizationPipeline, adapter_id
        )

    def __call__(self, inputs: str) -> List[Dict[str, str]]:
        """
        Args:
            inputs (:obj:`str`): a string to be summarized
        Return:
            A :obj:`list` of :obj:`dict` in the form of {"summary_text": "The string after summarization"}
        """
        return self.pipeline(inputs, truncation=True)
