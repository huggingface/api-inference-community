import json
import os
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "text2text-generation" not in ALLOWED_TASKS,
    "text2text-generation not implemented",
)
class TextToSpeechTestCase(TestCase):
    def setUp(self):
        model_id = TESTABLE_MODELS["text2text-generation"]
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = model_id
        os.environ["TASK"] = "text2text-generation"
        from app.main import app

        self.app = app

    @classmethod
    def setUpClass(cls):
        from app.main import get_pipeline

        get_pipeline.cache_clear()

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
        with TestClient(self.app) as client:
            response = client.post(
                "/",
                json={
                    "inputs": "English is tough. It can be understood "
                    "through thorough thought though."
                },
            )
        self.assertEqual(
            response.status_code,
            200,
        )
        result = json.loads(response.content)
        self.assertEqual(type(result), list)

        # Add more tests here
