import json
import logging
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Any

import joblib
import skops.io as sio
from app.pipelines import Pipeline
from huggingface_hub import snapshot_download


logger = logging.getLogger(__name__)
DEFAULT_FILENAME = "sklearn_model.joblib"


class SklearnBasePipeline(Pipeline):
    """Base class for sklearn-based inference pipelines

    Concrete implementations should add two methods:

    - `_get_output`: Method to generate model predictions
    - `__call__`: Should delegate to handle_call, add docstring and type
      annotations.

    """

    def __init__(self, model_id: str):
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
            config = dict()
            warnings.warn("`config.json` does not exist or is invalid.")

        self.model_file = (
            config.get("sklearn", {}).get("model", {}).get("file", DEFAULT_FILENAME)
        )
        self.model_format = config.get("sklearn", {}).get("model_format", "pickle")

        try:
            with warnings.catch_warnings(record=True) as record:
                if self.model_format == "pickle":

                    self.model = joblib.load(
                        open(Path(cached_folder) / self.model_file, "rb")
                    )
                elif self.model_format == "skops":
                    self.model = sio.load(
                        file=Path(cached_folder) / self.model_file, trusted=True
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

    @abstractmethod
    def _get_output(self, inputs: Any) -> Any:
        raise NotImplementedError(
            "Implement this method to get the model output (prediction)"
        )

    def __call__(self, inputs: Any) -> Any:
        """Handle call for getting the model prediction

        This method is responsible for handling all possible errors and
        warnings. To get the actual prediction, implement the `_get_output`
        method.

        The types of the inputs and output depend on the specific task being
        implemented.

        """

        if self._load_exception:
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
                res = self._get_output(inputs)
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
                    "warnings": _warnings,  # see issue #96
                }
                raise ValueError(json.dumps(error))
            else:
                # if there was an exception, we raise it so that routes.py can
                # catch and return a non 200 status code.
                raise exception

        if exception:
            raise exception

        return res
