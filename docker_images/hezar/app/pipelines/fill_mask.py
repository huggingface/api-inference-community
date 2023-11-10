from typing import Any, Dict, List

from hezar.models import Model

from app.pipelines import Pipeline


class FillMaskPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model = Model.load(model_id)

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
        model_outputs = self.model.predict(inputs)[0]

        outputs = []
        for fill_token in model_outputs:
            output = {
                "sequence": fill_token["sequence"],
                "score": fill_token["score"],
                "token": fill_token["token_id"],
                "token_str": fill_token["token"],
            }
            outputs.append(output)
        return outputs
