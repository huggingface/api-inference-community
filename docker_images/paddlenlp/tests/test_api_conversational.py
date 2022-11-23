import json
import os
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from parameterized import parameterized_class
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "conversational" not in ALLOWED_TASKS,
    "conversational not implemented",
)
@parameterized_class(
    [{"model_id": model_id} for model_id in TESTABLE_MODELS["conversational"]]
)
class ConversationalTestCase(TestCase):
    def setUp(self):
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = self.model_id
        os.environ["TASK"] = "fill-mask"
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

    def test_simple(self):
        first_round_inputs = {"text": "你好！"}
        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": first_round_inputs})

        self.assertEqual(
            response.status_code,
            200,
        )

        content = json.loads(response.content)
        self.assertEqual(type(content), dict)
        self.assertIn("generated_text", content)
        self.assertIn("conversation", content)
        self.assertIn("past_user_inputs", content["conversation"])
        self.assertIn("generated_responses", content["conversation"])
        self.assertEqual(len(content["conversation"]["generated_responses"]), 1)
        self.assertEqual(len(content["conversation"]["past_user_inputs"]), 1)

        second_round_inputs = {
            "text": "这是个测试",
            "past_user_inputs": content["conversation"]["past_user_inputs"],
            "generated_responses": content["conversation"]["generated_responses"],
        }

        with TestClient(self.app) as client:
            response = client.post("/", json=second_round_inputs)

        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)
        self.assertEqual(type(content), dict)
        self.assertIn("generated_text", content)
        self.assertIn("conversation", content)
        self.assertIn("past_user_inputs", content["conversation"])
        self.assertIn("generated_responses", content["conversation"])
        self.assertEqual(len(content["conversation"]["generated_responses"]), 2)
        self.assertEqual(len(content["conversation"]["past_user_inputs"]), 2)
