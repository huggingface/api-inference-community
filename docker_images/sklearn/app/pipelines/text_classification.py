import json
import logging
import warnings
from typing import Dict, List

from app.pipelines.common import SklearnBasePipeline


logger = logging.getLogger(__name__)


class TextClassificationPipeline(SklearnBasePipeline):
    def _get_output(self, inputs: str) -> List[Dict[str, float]]:
        res = []
        for i, c in enumerate(self.model.predict_proba([inputs]).tolist()[0]):
            res.append({"label": str(self.model.classes_[i]), "score": c})
        return [res]

    # even though we only call super, we implement this method to update the
    # docstring and the type annotation
    def __call__(self, inputs: str) -> List[Dict[str, float]]:
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`list`:. The object returned should be a list of one list like [[{"label": 0.9939950108528137}]] containing:
                - "label": A string representing what the label/class is. There can be multiple labels.
                - "score": A score between 0 and 1 describing how confident the model is for this label/class.
        """
        return super().handle_call(inputs)
