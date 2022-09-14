import os
import json
from unittest import TestCase, skipIf

from api_inference_community.validation import ffmpeg_read
from app.main import ALLOWED_TASKS
from parameterized import parameterized_class
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "grapheme-to-phoneme" not in ALLOWED_TASKS,
    "grapheme-to-phoneme not implemented",
)
@parameterized_class(
    [{"model_id": model_id} for model_id in TESTABLE_MODELS["grapheme-to-phoneme"]]
)
class TextToSpeechTestCase(TestCase):
    def setUp(self):
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = self.model_id
        os.environ["TASK"] = "grapheme-to-phoneme"
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
                    "through thorough thought though."})
        self.assertEqual(
            response.status_code,
            200,
        )
        result = json.loads(response.content)
    
        self.assertEqual(
            "IH NG G L IH SH   IH Z   T AH F   IH T   K AE N   B IY   "
            "AH N D ER S T UH D   TH R UW   TH ER OW   TH AO T   DH OW",
            result["phn"]
        )
