import json
import os
from pathlib import Path
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from common import SklearnTestCase
from parameterized import parameterized_class
from starlette.testclient import TestClient
from tests.test_api import TEST_CASES, TESTABLE_MODELS


@parameterized_class([{"test_case": x} for x in TESTABLE_MODELS["text-classification"]])
@skipIf(
    "text-classification" not in ALLOWED_TASKS,
    "text-classification not implemented",
)
class TextClassificationTestCase(SklearnTestCase, TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName=methodName)
        self.task = "text-classification"

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

    def test_malformed_question(self):
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
