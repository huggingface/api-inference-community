import json
import warnings
from pathlib import Path
from typing import Dict, List, Union

import joblib
from app.pipelines import Pipeline
from huggingface_hub import snapshot_download


DEFAULT_FILENAME = "sklearn_model.joblib"


class TabularDataPipeline(Pipeline):
    def __init__(self, model_id: str):
        cached_folder = snapshot_download(repo_id=model_id)
        try:
            with open(Path(cached_folder) / "config.json") as f:
                # this is the default path for configuration of a scikit-learn
                # project. If the project is created using `skops`, it should have
                # this file.
                config = json.load(f)
            model_file = config["sklearn"]["model"]["file"]
        except Exception as e:
            print(e)
            # If for whatever reason we fail to detect requirements of the project,
            # we install the latest scikit-learn.
            # TODO: we should log this, or warn or something.
            model_file = DEFAULT_FILENAME

        self.model = joblib.load(open(Path(cached_folder) / model_file, "rb"))

    def __call__(
        self, inputs: Dict[str, Dict[str, List[Union[str, float]]]]
    ) -> List[Union[str, float]]:
        """
        Args:
            inputs (:obj:`dict`):
                a dictionary containing a key 'data' mapping to a dict in which
                the values represent each column.
        Return:
            A :obj:`list` of floats or strings: The classification output for each row.
        """
        column_values = list(inputs["data"].values())
        rows = list(zip(*column_values))
        # TODO: we should do sth with the warnings.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return self.model.predict(rows).tolist()
