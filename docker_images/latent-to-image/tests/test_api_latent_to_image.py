import json
import os
from io import BytesIO
from unittest import TestCase, skipIf

import PIL
import torch
from app.main import ALLOWED_TASKS
from parameterized import parameterized_class
from safetensors.torch import _tobytes
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "latent-to-image" not in ALLOWED_TASKS,
    "latent-to-image not implemented",
)
@parameterized_class(
    [{"model_id": model_id} for model_id in TESTABLE_MODELS["latent-to-image"]]
)
class TextToImageTestCase(TestCase):
    def setUp(self):
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = self.model_id
        os.environ["TASK"] = "latent-to-image"
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
        inputs = torch.randn([1, 4, 64, 64], generator=torch.Generator().manual_seed(0))
        shape = json.dumps(list(inputs.shape))
        dtype = str(inputs.dtype).split(".")[-1]
        tensor_data = _tobytes(inputs, "inputs")
        headers = {"shape": shape, "dtype": dtype}

        with TestClient(self.app) as client:
            response = client.post("/", data=tensor_data, headers=headers)

        self.assertEqual(
            response.status_code,
            200,
        )

        image = PIL.Image.open(BytesIO(response.content))
        self.assertTrue(isinstance(image, PIL.Image.Image))

    def test_malformed_input(self):
        headers = {"shape": json.dumps([1, 4, 64, 64]), "dtype": "float32"}
        with TestClient(self.app) as client:
            response = client.post("/", data=b"\xc3\x28", headers=headers)

        self.assertEqual(
            response.status_code,
            400,
        )
        self.assertEqual(
            response.content,
            b'{"error":"buffer length (2 bytes) after offset (0 bytes) must be a multiple of element size (4)"}',
        )
