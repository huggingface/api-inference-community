from http.client import responses
from typing import Any, Dict, List, Union

from app.pipelines import Pipeline
from paddlenlp.taskflow import Taskflow


class FillMaskPipeline(Pipeline):
    def __init__(self, model_id: str):
        # TODO: how to setup params
        self.taskflow = Taskflow("fill_mask", task_path=model_id, from_hf_hub=True)
        

    def __call__(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """
        Args:
            inputs (:obj:`dict`): ???
        Return:
            A :obj:`dict`:. ???
        """
        
        ## To be implemented once the API clears up



        