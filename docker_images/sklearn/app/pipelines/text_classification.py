import json
import logging
import warnings
from typing import Dict, List

from app.pipelines.common import SklearnBasePipeline


logger = logging.getLogger(__name__)


class TextClassificationPipeline(SklearnBasePipeline):
    def __init__(self, model_id: str):
        super().__init__(model_id)

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
        if self._load_exception:
            # there has been an error while loading the model. We need to raise
            # that, and can't call predict on the model.
            raise ValueError(
                "An error occurred while loading the model: "
                f"{str(self._load_exception)}"
            )

        _warnings = []
        exception = None
        try:
            with warnings.catch_warnings(record=True) as record:
                # We will predict probabilities for each class and return them as
                # list of list of dictionaries
                # below is a numpy array of probabilities of each class
                res = []
                for i, c in enumerate(self.model.predict_proba([inputs]).tolist()[0]):
                    res.append({"label": str(self.model.classes_[i]), "score": c})
                res = [res]
        except Exception as e:
            exception = e

        for warning in record:
            _warnings.append(f"{warning.category.__name__}({warning.message})")

        for warning in self._load_warnings:
            _warnings.append(f"{warning.category.__name__}({warning.message})")

        if _warnings:
            for warning in _warnings:
                logger.warning(warning)

            if not exception:
                # we raise an error if there are any warnings, so that routes.py
                # can catch and return a non 200 status code.
                error = {
                    "error": "There were warnings while running the model.",
                    "output": res,
                }
                raise ValueError(json.dumps(error))
            else:
                # if there was an exception, we raise it so that routes.py can
                # catch and return a non 200 status code.
                raise exception

        return res
