import json
import warnings
from pathlib import Path

import joblib
from app.pipelines import Pipeline
from huggingface_hub import snapshot_download


DEFAULT_FILENAME = "sklearn_model.joblib"


class SklearnBasePipeline(Pipeline):
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
