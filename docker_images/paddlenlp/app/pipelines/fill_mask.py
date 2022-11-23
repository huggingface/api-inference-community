from typing import Any, Dict, List

from app.pipelines import Pipeline
from paddlenlp.taskflow import Taskflow


class FillMaskPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.taskflow = Taskflow("fill_mask", task_path=model_id, from_hf_hub=True)

    def __call__(self, inputs: str) -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`str`): a string to be filled from, must contain the [MASK] token (check model card for exact name of the mask)
        Return:
            A :obj:`list`:. The object returned should be like [{"entity_group": "XXX", "word": "some word", "start": 3, "end": 6, "score": 0.82}] containing :
                - "sequence": The actual sequence of tokens that ran against the model (may contain special tokens)
                - "score": The probability for this token.
                - "token": The id of the token
                - "token_str": The string representation of the token
        """
        return self.taskflow(inputs)
