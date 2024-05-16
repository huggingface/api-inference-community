import json
import os
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "token-classification" not in ALLOWED_TASKS,
    "token-classification not implemented",
)
class TokenClassificationTestCase(TestCase):
    def setUp(self):
        model_id = TESTABLE_MODELS["token-classification"]
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = model_id
        os.environ["TASK"] = "token-classification"
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
        inputs = "Le patient prend du paracétamol à 500mg, 3 fois par jour. "

        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": inputs})

        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)
        self.assertEqual(type(content), list)
        self.assertEqual(
            set(k for el in content for k in el.keys()),
            {"entity_group", "word", "start", "end", "score", "value"},
        )

        with TestClient(self.app) as client:
            response = client.post("/", json=inputs)

        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)
        self.assertEqual(type(content), list)
        self.assertEqual(
            set(k for el in content for k in el.keys()),
            {"entity_group", "word", "start", "end", "score", "value"},
        )

    def test_formatted(self):
        inputs = "Hello, my name is [John](PER) and I live in [New York](LOC)"

        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": inputs})

        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)
        self.assertEqual(type(content), list)
        self.assertEqual(
            set(k for el in content for k in el.keys()),
            {"entity_group", "word", "start", "end", "score", "value"},
        )
        entity_groups = {el["entity_group"] for el in content}
        self.assertGreaterEqual(entity_groups, {"PER", "LOC"})

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
