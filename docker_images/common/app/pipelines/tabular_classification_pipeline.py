from typing import Dict, List, Union

from app.pipelines import Pipeline


class TabularClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        raise NotImplementedError(
            "Please implement TabularClassificationPipeline __init__ function"
        )

    def __call__(
        self, inputs: Dict[str, Dict[str, List[Union[int, str, float]]]]
    ) -> List[Union[int, str, float]]:
        """
        Args:
            inputs (:obj:`dict`):
                a dictionary containing a key 'data' mapping to a dict in which
                the values represent each column.
        Return:
            A :obj:`list` of int, str, or float: The classification output for each row.
        """
        # IMPLEMENT_THIS
        raise NotImplementedError(
            "Please implement TabularClassificationPipeline __init__ function"
        )
