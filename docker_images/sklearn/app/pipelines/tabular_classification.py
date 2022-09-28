from typing import Dict, List, Union

import pandas as pd
from app.pipelines.common import SklearnBasePipeline


class TabularClassificationPipeline(SklearnBasePipeline):
    def _get_output(
        self, inputs: Dict[str, Dict[str, List[Union[str, float]]]]
    ) -> List[Union[str, float]]:
        # We convert the inputs to a pandas DataFrame, and use self.columns
        # to order the columns in the order they're expected, ignore extra
        # columns given if any, and put NaN for missing columns.
        data = pd.DataFrame(inputs["data"], columns=self.columns)
        res = self.model.predict(data).tolist()
        return res

    # even though we only delegate the call, we implement this method have the
    # correct docstring and type annotations
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
        return self.handle_call(inputs)
