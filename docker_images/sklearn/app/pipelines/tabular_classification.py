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
