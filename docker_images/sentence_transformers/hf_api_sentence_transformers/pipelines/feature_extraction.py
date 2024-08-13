import os
from typing import List

from hf_api_sentence_transformers.pipelines import Pipeline
from sentence_transformers import SentenceTransformer


class FeatureExtractionPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        self.model = SentenceTransformer(
            model_id, use_auth_token=os.getenv("HF_API_TOKEN")
        )

    def __call__(self, inputs: str) -> List[float]:
        """
        Args:
            inputs (:obj:`str`):
                a string to get the features of.
        Return:
            A :obj:`list` of floats: The features computed by the model.
        """
        return self.model.encode(inputs).tolist()
