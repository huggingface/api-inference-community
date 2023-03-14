import json
import os
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from parameterized import parameterized_class
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "text-classification" not in ALLOWED_TASKS,
    "text-classification not implemented",
)
@parameterized_class(
    [{"model_id": model_id} for model_id in TESTABLE_MODELS["text-classification"]]
)
class TextClassificationTestCase(TestCase):
    def setUp(self):
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = self.model_id
        os.environ["TASK"] = "text-classification"
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
        inputs = "beautiful"

        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": inputs})
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

        with TestClient(self.app) as client:
            response = client.post("/", json=inputs)

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

    def test_multiple_words(self):
        inputs = "this is great"

        # For "language-identification" substask, fasttext can identify the language of a sentence
        # but when getting a word vector's nearest neighbors, only a single word is valid as an input
        expected_status_code = (
            200 if "language-identification" in self.model_id else 400
        )
        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": inputs})
        self.assertEqual(
            response.status_code,
            expected_status_code,
        )
