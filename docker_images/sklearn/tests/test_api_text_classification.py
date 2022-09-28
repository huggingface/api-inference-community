import json
import os
from pathlib import Path
from unittest import TestCase, skipIf

import pytest
from app.main import ALLOWED_TASKS
from parameterized import parameterized_class
from starlette.testclient import TestClient
from tests.test_api import TEST_CASES, TESTABLE_MODELS


@parameterized_class([{"test_case": x} for x in TESTABLE_MODELS["text-classification"]])
@skipIf(
    "text-classification" not in ALLOWED_TASKS,
    "text-classification not implemented",
)
class TextClassificationTestCase(TestCase):
    def setUp(self):
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = self.test_case
        os.environ["TASK"] = "text-classification"
        self.case_data = TEST_CASES["text-classification"][self.test_case]

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
            response = client.post("/", json={"inputs": data["data"][0]})
        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)
        self.assertEqual(type(content), list)
        self.assertEqual(len(content), 1)
        self.assertEqual(type(content[0]), list)
        self.assertEqual(
            set(k for el in content[0] for k in el.keys()),
            {"label", "score"},
        )
        self.assertEqual(len(content), expected_output_len)

    def test_wrong_sklearn_version_warning(self):
        # if the wrong sklearn version is used the model will be loaded and
        # gives an output, but warnings are raised. This test makes sure the
        # right warnings are raised and that the output is included in the
        # error message.
        self._check_requirement(self.case_data["old_sklearn"] and self._can_load())

        data = self.data
        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": data["data"][0]})

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
        assert error_message["output"] == self.expected_output

    def test_cannot_load_model(self):
        # test the error message when the model cannot be loaded on a wrong
        # sklearn version
        self._check_requirement(not self.case_data["loads_on_new_sklearn"])

        data = self.data
        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": data["data"][0]})

        assert response.status_code == 400
        content = json.loads(response.content)
        assert "error" in content
        assert "An error occurred while loading the model:" in content["error"]

    def test_malformed_question(self):
        # testing wrong input for inference API
        with TestClient(self.app) as client:
            response = client.post("/", data=b"\xc3\x28")

        self.assertEqual(
            response.status_code,
            400,
        )
        self.assertEqual(
            response.content,
            b'{"error":"\'utf-8\' codec can\'t decode byte 0xc3 in position 0: invalid continuation byte"}',
        )
