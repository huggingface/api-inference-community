from typing import Dict, List, Union

import joblib
from app.pipelines import Pipeline
from huggingface_hub import cached_download, hf_hub_url


DEFAULT_FILENAME = "sklearn_model.joblib"


class TabularDataPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model = joblib.load(
            open(cached_download(hf_hub_url(model_id, DEFAULT_FILENAME)), "rb")
        )

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
        return self.model.predict(rows).tolist()
