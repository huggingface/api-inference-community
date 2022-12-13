from typing import Dict, List

from app.pipelines import Pipeline
from paddlenlp.taskflow import Taskflow


class SummarizationPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.taskflow = Taskflow(
            "text_summarization", task_path=model_id, from_hf_hub=True
        )

    def __call__(self, inputs: str) -> List[Dict[str, str]]:
        """
        Args:
            inputs (:obj:`str`): a string to be summarized
        Return:
            A :obj:`list` of :obj:`dict` in the form of {"summary_text": "The string after summarization"}
        """
        results = self.taskflow(inputs)
        return [{"summary_text": results[0]}]
