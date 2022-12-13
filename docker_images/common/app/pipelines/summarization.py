from typing import Dict, List

from app.pipelines import Pipeline


class SummarizationPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        raise NotImplementedError(
            "Please implement SummarizationPipeline __init__ function"
        )

    def __call__(self, inputs: str) -> List[Dict[str, str]]:
        """
        Args:
            inputs (:obj:`str`): a string to be summarized
        Return:
            A :obj:`list` of :obj:`dict` in the form of {"summary_text": "The string after summarization"}
        """
        # IMPLEMENT_THIS
        raise NotImplementedError(
            "Please implement SummarizationPipeline __init__ function"
        )
