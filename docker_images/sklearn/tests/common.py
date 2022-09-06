import json
import os
from pathlib import Path
from unittest import TestCase

import pytest
from starlette.testclient import TestClient
from tests.test_api import TEST_CASES


class SklearnTestCase(TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName=methodName)

    def setUp(self, **kwargs):
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        self.test_case = "test_case"
        self.task = "task"
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
        # this test is task specific success case
        pass

    def test_wrong_sklearn_version_warning(self):
        # if the wrong sklearn version is used the model will be loaded and
        # gives an output, but warnings are raised. This test makes sure the
        # right warnings are raised and that the output is included in the
        # error message.
        self._check_requirement(self.case_data["old_sklearn"] and self._can_load())

        data = self.data
        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": data})

        assert response.status_code == 400
        content = json.loads(response.content)
        assert "error" in content
        assert "warnings" in content
        assert any("Trying to unpickle estimator" in w for w in content["warnings"])
        error_message = json.loads(content["error"])
        assert error_message["output"] == self.expected_output

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
