from typing import Any, Dict, List

from app.pipelines import Pipeline
from sentence_transformers import SentenceTransformer

class FeatureExtractionPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        self.model = SentenceTransformer(model_id)

    def __call__(self, inputs: str) -> List[float]:
        """
        Args:
            inputs (:obj:`List[str]`):
                a string to get the features of.
        Return:
            A :obj:`list` of floats: The features computed by the model.
        """
        return self.model.encode(inputs).tolist()
