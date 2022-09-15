import json
import logging
import warnings
from typing import Dict, List, Union

import pandas as pd
from app.pipelines.common import SklearnBasePipeline


logger = logging.getLogger(__name__)


class TabularClassificationPipeline(SklearnBasePipeline):
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
