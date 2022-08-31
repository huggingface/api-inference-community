import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List

import joblib
from app.pipelines import Pipeline
from huggingface_hub import snapshot_download


DEFAULT_FILENAME = "sklearn_model.joblib"

logger = logging.getLogger(__name__)


class TextClassificationPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        cached_folder = snapshot_download(repo_id=model_id)
        self._load_warnings = []
        self._load_exception = None
        try:
            with open(Path(cached_folder) / "config.json") as f:
                # this is the default path for configuration of a scikit-learn
                # project. If the project is created using `skops`, it should have
                # this file.
                config = json.load(f)
        except Exception:
            # If for whatever reason we fail to detect requirements of the
            # project, we install the latest scikit-learn.
            config = dict()

        self.model_file = (
            config.get("sklearn", {}).get("model", {}).get("file", DEFAULT_FILENAME)
        )

        try:
            with warnings.catch_warnings(record=True) as record:
                self.model = joblib.load(
                    open(Path(cached_folder) / self.model_file, "rb")
                )
                if len(record) > 0:
                    # if there's a warning while loading the model, we save it so
                    # that it can be raised to the user when __call__ is called.
                    self._load_warnings += record
        except Exception as e:
            # if there is an exception while loading the model, we save it to
            # raise the write error when __call__ is called.
            self._load_exception = e
        # use labels from config file if available
        self.labels = config.get("sklearn", {}).get("labels", None)
        if not self.labels:
            self.labels = self.model.classes_

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
                prob = self.model.predict_proba([inputs["data"]]).tolist()
                res = []
                for i, c in enumerate(prob[0]):
                    res.append({"label": str(self.labels[i]), "score": c})
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
