import json
import os
from base64 import b64encode
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from parameterized import parameterized_class
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "zero-shot-image-classification" not in ALLOWED_TASKS,
    "zero-shot-image-classification not implemented",
)
@parameterized_class(
    [
        {"model_id": model_id}
        for model_id in TESTABLE_MODELS["zero-shot-image-classification"]
    ]
)
class ZeroShotImageClassificationTestCase(TestCase):
    def setUp(self):
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = self.model_id
        os.environ["TASK"] = "zero-shot-image-classification"
        from app.main import app, get_pipeline

        get_pipeline.cache_clear()

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

    def read(self, filename: str) -> bytes:
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, "samples", filename)
        with open(filename, "rb") as f:
            bpayload = f.read()
        return bpayload

    def test_simple(self):
        input_dict = {
            "inputs": b64encode(self.read("plane.jpg")).decode("utf-8"),
            "parameters": {
                "candidate_labels": [
                    "airplane",
                    "superman",
                    "crumpet",
                ],
            },
        }

        with TestClient(self.app) as client:
            response = client.post("/", json=input_dict)

        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)
        self.assertEqual(type(content), list)
        self.assertEqual(set(type(el) for el in content), {dict})
        self.assertEqual(
            set((k, type(v)) for el in content for (k, v) in el.items()),
            {("label", str), ("score", float)},
        )
        res = {e["label"]: e["score"] for e in content}
        self.assertGreater(res["airplane"], 0.9)

    def test_different_resolution(self):
        input_dict = {
            "inputs": b64encode(self.read("plane2.jpg")).decode("utf-8"),
            "parameters": {
                "candidate_labels": [
                    "airplane",
                    "superman",
                    "crumpet",
                ],
            },
        }

        with TestClient(self.app) as client:
            response = client.post("/", json=input_dict)

        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)
        self.assertEqual(type(content), list)
        self.assertEqual(set(type(el) for el in content), {dict})
        self.assertEqual(
            set(k for el in content for k in el.keys()), {"label", "score"}
        )
        res = {e["label"]: e["score"] for e in content}
        self.assertGreater(res["airplane"], 0.9)
