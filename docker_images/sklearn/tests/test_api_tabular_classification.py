import json
import os
from pathlib import Path
from unittest import TestCase, skipIf

import pytest
from app.main import ALLOWED_TASKS
from parameterized import parameterized_class
from starlette.testclient import TestClient
from tests.test_api import TEST_CASES, TESTABLE_MODELS


@parameterized_class(
    [{"test_case": x} for x in TESTABLE_MODELS["tabular-classification"]]
)
@skipIf(
    "tabular-classification" not in ALLOWED_TASKS,
    "tabular-classification not implemented",
)
class TabularClassificationTestCase(TestCase):
    # self.repo_id and self.input are provided by parameterized_class
    def setUp(self):
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = self.test_case
        os.environ["TASK"] = "tabular-classification"

        from app.main import app

        self.case_data = TEST_CASES["tabular-classification"][self.test_case]

        self.app = app
        sample_folder = Path(__file__).parent / "generators" / "samples"
        self.data = json.load(open(sample_folder / self.case_data["input"], "r"))
        self.expected_output = json.load(
            open(sample_folder / self.case_data["output"], "r")
        )

    def tearDown(self):
        if self.old_model_id is not None:
            os.environ["MODEL_ID"] = self.old_model_id
        else:
            del os.environ["MODEL_ID"]
        if self.old_task is not None:
            os.environ["TASK"] = self.old_task
        else:
            del os.environ["TASK"]

    def _check_requirement(self, requirement):
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

    def test_malformed_input(self):
        with TestClient(self.app) as client:
            response = client.post("/", data=b"Where do I live ?")

        self.assertEqual(
            response.status_code,
            400,
        )
        content = json.loads(response.content)
        self.assertEqual(set(content.keys()), {"error"})

    def test_missing_columns(self):
        data = self.data["data"].copy()
        data.pop(next(iter(data.keys())))

        inputs = {"data": data}
        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": inputs})

        self.assertEqual(
            response.status_code,
            400,
        )
