import base64
import os
from io import BytesIO
from unittest import TestCase, skipIf

import PIL
from app.main import ALLOWED_TASKS
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "image-to-image" not in ALLOWED_TASKS,
    "image-to-image not implemented",
)
class ImageToImageTestCase(TestCase):
    def setUp(self):
        model_id = TESTABLE_MODELS["image-to-image"]
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = model_id
        os.environ["TASK"] = "image-to-image"
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
        image = PIL.Image.new("RGB", (64, 64))
        parameters = {"prompt": "soap bubble"}

        with TestClient(self.app) as client:
            response = client.post(
                "/",
                json={
                    "image": base64.b64encode(image).decode("utf-8"),
                    "parameters": parameters,
                },
            )

        self.assertEqual(
            response.status_code,
            200,
        )

        image = PIL.Image.open(BytesIO(response.content))
        self.assertTrue(isinstance(image, PIL.Image.Image))

    def test_malformed_input(self):
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
