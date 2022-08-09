import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Union

import joblib
import pandas as pd
from app.pipelines import Pipeline
from huggingface_hub import snapshot_download


DEFAULT_FILENAME = "sklearn_model.joblib"

logger = logging.getLogger()


class TabularClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        cached_folder = snapshot_download(repo_id=model_id)
        self._load_warnings = []
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
        # use column names from the config file if available, to give the data
        # to the model in the right order.
        self.columns = config.get("sklearn", {}).get("columns", None)

    def __call__(
        self, inputs: Dict[str, Dict[str, List[Union[str, float]]]]
    ) -> List[Union[str, float]]:
        """Return the output of the model for given inputs.

        Args:
            inputs (:obj:`dict`):
                a dictionary containing a key 'data' mapping to a dict in which
                the values represent each column.
        Return:
            A :obj:`list` of floats or strings: The classification output for
            each row.
        """
        if getattr(self, "_load_exception", None):
            # there has been an error while loading the model. We need to raise
            # that, and can't call predict on the model.
            raise ValueError(
                "An error occurred while loading the model: "
                f"{str(self._load_exception)}"
            )

        _warnings = []
        if self.columns:
            # TODO: we should probably warn if columns are not configured, we
            # really do need them.
            given_cols = set(inputs["data"].keys())
            expected = set(self.columns)
            extra = given_cols - expected
            missing = expected - given_cols
            if extra:
                _warnings.append(
                    f"The following columns were given but not expected: {extra}"
                )

            if missing:
                _warnings.append(
                    f"The following columns were expected but not given: {missing}"
                )

        exception = None
        try:
            with warnings.catch_warnings(record=True) as record:
                # We convert the inputs to a pandas DataFrame, and use self.columns
                # to order the columns in the order they're expected, ignore extra
                # columns given if any, and put NaN for missing columns.
                data = pd.DataFrame(inputs["data"], columns=self.columns)
                res = self.model.predict(data).tolist()
        except Exception as e:
            exception = e

        for warning in record:
            _warnings.append(f"{warning.category.__name__}({warning.message})")

        for warning in self._load_warnings:
            _warnings.append(f"{warning.category.__name__}({warning.message})")

        # making sure warnings are recorded only once.
        _warnings = set(_warnings)
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
