import json
import os
from pathlib import Path
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from common import SklearnTestCase
from parameterized import parameterized, parameterized_class
from starlette.testclient import TestClient
from tests.test_api import TEST_CASES, TESTABLE_MODELS


@parameterized_class(
    [{"test_case": x} for x in TESTABLE_MODELS["tabular-classification"]]
)
@skipIf(
    "tabular-classification" not in ALLOWED_TASKS,
    "tabular-classification not implemented",
)
class TabularClassificationTestCase(SklearnTestCase, TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName=methodName)
        self.task = "tabular-classification"

    def setUp(self):
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = self.test_case
        os.environ["TASK"] = self.task

        self.case_data = TEST_CASES[self.task][self.test_case]
        sample_folder = Path(__file__).parent / "generators" / "samples"
        self.data = json.load(open(sample_folder / self.case_data["input"], "r"))
        self.expected_output = json.load(
            open(sample_folder / self.case_data["output"], "r")
        )
        from app.main import app

        self.app = app

    def test_success_code(self):
        # This test does a sanity check on the output and checks the response
        # code which should be 200. This requires the model to be from the
        # latest sklearn which is the one installed locally.
        self._check_requirement(not self.case_data["old_sklearn"])

        data = self.data
        expected_output_len = len(self.expected_output)

        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": data})

        assert response.status_code == 200
        content = json.loads(response.content)
        assert type(content) == list
        assert len(content) == expected_output_len

    @parameterized.expand(
        [
            (["add"], ["The following columns were given but not expected:"]),
            (["drop"], ["The following columns were expected but not given:"]),
            (
                ["add", "drop"],
                [
                    "The following columns were given but not expected:",
                    "The following columns were expected but not given:",
                ],
            ),
        ]
    )
    def test_extra_columns(self, column_operations, warn_messages):
        # Test that the right warning is raised when there are extra columns in
        # the input.
        self._check_requirement(self.case_data["has_config"] and self._can_load())

        data = self.data.copy()
        if "drop" in column_operations:
            # we remove the first column in the data. Note that `data` is a
            # dict of column names to values.
            data["data"].pop(next(iter(data["data"].keys())))
        if "add" in column_operations:
            # we add an extra column to the data, the same as the first column.
            # Note that `data` is a dict of column names to values.
            data["data"]["extra_column"] = next(iter(data["data"].values()))

        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": data})

        assert response.status_code == 400
        content = json.loads(response.content)
        assert "error" in content
        assert "warnings" in content

        for warn_message in warn_messages:
            assert any(warn_message in w for w in content["warnings"])

        if "drop" not in column_operations or self.case_data["accepts_nan"]:
            # the predict does not raise an error
            error_message = json.loads(content["error"])
            assert len(error_message["output"]) == len(self.expected_output)
            if "drop" not in column_operations:
                # if no column was dropped, the predictions should be the same
                assert error_message["output"] == self.expected_output
        else:
            # otherwise some columns will be empty and predict errors.
            assert (
                "does not accept missing values encoded as NaN natively"
                in content["error"]
            )

    def test_malformed_input(self):
        self._check_requirement(self._can_load())

        with TestClient(self.app) as client:
            response = client.post("/", data=b"Where do I live ?")

        assert response.status_code == 400
        content = json.loads(response.content)
        assert set(content.keys()) == {"error"}
