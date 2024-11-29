import json
import os
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from parameterized import parameterized_class
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "voice-activity-detection" not in ALLOWED_TASKS,
    "voice-activity-detection not implemented",
)

@parameterized_class(
    [{"model_id": model_id} for model_id in TESTABLE_MODELS["voice-activity-detection"]]
)


class VoiceActivityDetectionTestCase(TestCase):
    def setUp(self):
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = self.model_id
        os.environ["TASK"] = "voice-activity-detection"
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

    def read(self, filename: str) -> bytes:
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, "samples", filename)
        with open(filename, "rb") as f:
            bpayload = f.read()
        return bpayload

    def test_simple(self):
        bpayload = self.read("sample1.flac")

        with TestClient(self.app) as client:
            response = client.post("/", data=bpayload)

        self.assertEqual(
            response.status_code,
            200,
        )

        content = json.loads(response.content)
        self.assertEqual(type(content), list)
        self.assertEqual(type(content[0]), dict)


    def test_malformed_audio(self):
        bpayload = self.read("malformed.flac")

        with TestClient(self.app) as client:
            response = client.post("/", data=bpayload)

        self.assertEqual(
            response.status_code,
            400,
        )
        self.assertEqual(response.content, b'{"error":"Malformed soundfile"}')

    def test_dual_channel_audiofile(self):
        bpayload = self.read("sample1_dual.ogg")

        with TestClient(self.app) as client:
            response = client.post("/", data=bpayload)

        self.assertEqual(
            response.status_code,
            200,
        )

        content = json.loads(response.content)
        self.assertEqual(type(content), list)
        self.assertEqual(type(content[0]), dict)


    def test_webm_audiofile(self):
        bpayload = self.read("sample1.webm")

        with TestClient(self.app) as client:
            response = client.post("/", data=bpayload)

        self.assertEqual(
            response.status_code,
            200,
        )

        content = json.loads(response.content)
        self.assertEqual(type(content), list)
        self.assertEqual(type(content[0]), dict)
