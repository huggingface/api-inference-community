"""Tests for tabular regression

The test class is almost completely copied from TabularClassificationTestCase,
only changing to different parametrized test cases.

"""

import json
import os
from pathlib import Path
from unittest import TestCase, skipIf

import pytest
from app.main import ALLOWED_TASKS
from parameterized import parameterized, parameterized_class
from starlette.testclient import TestClient
from tests.test_api import TEST_CASES, TESTABLE_MODELS


@parameterized_class([{"test_case": x} for x in TESTABLE_MODELS["tabular-regression"]])
@skipIf(
    "tabular-regression" not in ALLOWED_TASKS,
    "tabular-regression not implemented",
)
class TabularRegressionTestCase(TestCase):
    # self.test_case is provided by parameterized_class
    def setUp(self):
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = self.test_case
        os.environ["TASK"] = "tabular-regression"

        self.case_data = TEST_CASES["tabular-regression"][self.test_case]

        sample_folder = Path(__file__).parent / "generators" / "samples"
        self.data = json.load(open(sample_folder / self.case_data["input"], "r"))
        self.expected_output = json.load(
            open(sample_folder / self.case_data["output"], "r")
        )

        from app.main import app

        self.app = app

    def tearDown(self):
        if self.old_model_id is not None:
            os.environ["MODEL_ID"] = self.old_model_id
        else:
            del os.environ["MODEL_ID"]
        if self.old_task is not None:
            os.environ["TASK"] = self.old_task
        else:
            del os.environ["TASK"]

    def _can_load(self):
        # to load a model, it has to either support being loaded on new sklearn
        # versions, or it needs to be saved by a new sklearn version, since the
        # assumption is that the current sklearn version is the latest.
        return (
            self.case_data["loads_on_new_sklearn"] or not self.case_data["old_sklearn"]
        )

    def _check_requirement(self, requirement):
        # This test is not supposed to run and is thus skipped.
        if not requirement:
            pytest.skip("Skipping test because requirements are not met.")

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

    def test_wrong_sklearn_version_warning(self):
        # if the wrong sklearn version is used the model will be loaded and
        # gives an output, but warnings are raised. This test makes sure the
        # right warnings are raised and that the output is included in the
        # error message.
        self._check_requirement(self.case_data["old_sklearn"] and self._can_load())

        data = self.data
        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": data})

        # check response
        assert response.status_code == 400
        content = json.loads(response.content)
        assert "error" in content
        assert "warnings" in content

        # check warnings
        assert any("Trying to unpickle estimator" in w for w in content["warnings"])
        warnings = json.loads(content["error"])["warnings"]
        assert any("Trying to unpickle estimator" in w for w in warnings)

        # check error
        error_message = json.loads(content["error"])
        assert len(error_message["output"]) == len(self.expected_output)
        for val_output, val_expected in zip(
            error_message["output"], self.expected_output
        ):
            self.assertAlmostEqual(val_output, val_expected)

    def test_cannot_load_model(self):
        # test the error message when the model cannot be loaded on a wrong
        # sklearn version
        self._check_requirement(not self.case_data["loads_on_new_sklearn"])

        data = self.data
        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": data})

        assert response.status_code == 400
        content = json.loads(response.content)
        assert "error" in content
        assert "An error occurred while loading the model:" in content["error"]

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
            # predict does not raise an error
            error_message = json.loads(content["error"])
            assert len(error_message["output"]) == len(self.expected_output)
            if "drop" not in column_operations:
                # if no column was dropped, the predictions should be the same
                for val_output, val_expected in zip(
                    error_message["output"], self.expected_output
                ):
                    self.assertAlmostEqual(val_output, val_expected)
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
