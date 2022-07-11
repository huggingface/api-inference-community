import json
import os
from pathlib import Path
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "tabular-classification" not in ALLOWED_TASKS,
    "tabular-classification not implemented",
)
class TabularClassificationTestCase(TestCase):
    def setUp(self):
        test_case = TESTABLE_MODELS["tabular-classification"]
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = test_case["repo_id"]
        os.environ["TASK"] = "tabular-classification"

        from app.main import app

        self.app = app
        self.test_data = test_case["input"]
        self.data = json.load(
            open(Path(os.path.dirname(__file__)) / "samples" / self.test_data, "r")
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

    def test_simple(self):
        data = self.data
        expected_output_len = len(next(iter(data["data"].values())))

        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": data})
        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)
        self.assertEqual(type(content), list)
        self.assertEqual(len(content), expected_output_len)

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
