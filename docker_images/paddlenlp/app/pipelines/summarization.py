from typing import Dict, List, Union

from app.pipelines import Pipeline
from paddlenlp.taskflow import Taskflow


class SummarizationPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.taskflow = Taskflow(
            "text_summarization", task_path=model_id, from_hf_hub=True
        )

    def __call__(self, inputs: Union[str, List[str]]) -> List[Dict[str, str]]:
        """
        Args:
            inputs (:obj:`str`): a string to be summarized
        Return:
            A :obj:`dict`:. The object return should be like {"summarization_text": "The string after summarization"}
        """
        results = self.taskflow(inputs)
        return [{"summarization_text": result} for result in results]
