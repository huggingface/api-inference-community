from typing import Any, Dict, List

from app.pipelines import Pipeline


class FillMaskPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        raise NotImplementedError("Please implement FillMaskPipeline __init__ function")

    def __call__(self, inputs: str) -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`str`): a string to be filled from, must contain one and only one [MASK] token (check model card for exact name of the mask)
        Return:
            A :obj:`list`:. a list of dicts containing the following:
                - "sequence": The actual sequence of tokens that ran against the model (may contain special tokens)
                - "score": The probability for this token.
                - "token": The id of the token
                - "token_str": The string representation of the token
        """
        # IMPLEMENT_THIS
        raise NotImplementedError("Please implement FillMaskPipeline __call__ function")
