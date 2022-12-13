import json
import os
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from parameterized import parameterized_class
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "summarization" not in ALLOWED_TASKS,
    "summarization not implemented",
)
@parameterized_class(
    [{"model_id": model_id} for model_id in TESTABLE_MODELS["summarization"]]
)
class SummarizationTestCase(TestCase):
    def setUp(self):
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = self.model_id
        os.environ["TASK"] = "summarization"
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

    def test_single_input(self):
        text = "test"
        with TestClient(self.app) as client:
            response = client.post("/", json=text)

        self.assertEqual(
            response.status_code,
            200,
        )

        content = json.loads(response.content)
        self.assertEqual(type(content), list)
        self.assertEqual(len(content), 1)
        for result in content:
            self.assertIn("summary_text", result)
