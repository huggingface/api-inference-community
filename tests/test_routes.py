import io
import json
import logging
import os
from unittest import TestCase

import numpy as np
from api_inference_community.routes import pipeline_route, status_ok
from PIL import Image
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient


class ValidationTestCase(TestCase):
    def read(self, filename: str) -> bytes:
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, "samples", filename)
        with open(filename, "rb") as f:
            bpayload = f.read()
        return bpayload

    def test_pipeline(self):
        os.environ["TASK"] = "text-classification"

        class Pipeline:
            def __init__(self):
                pass

            def __call__(self, input_: str):
                return {"some": "json serializable"}

        def get_pipeline():
            return Pipeline()

        routes = [
            Route("/{whatever:path}", status_ok),
            Route("/{whatever:path}", pipeline_route, methods=["POST"]),
        ]

        app = Starlette(routes=routes)

        @app.on_event("startup")
        async def startup_event():
            logger = logging.getLogger("uvicorn.access")
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.handlers = [handler]

            # Link between `api-inference-community` and framework code.
            app.get_pipeline = get_pipeline

        with TestClient(app) as client:
            response = client.post("/", data=b"Some")
        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(response.headers["x-compute-characters"], "4")
        self.assertEqual(response.content, b'{"some":"json serializable"}')

    def test_invalid_pipeline(self):
        os.environ["TASK"] = "invalid"

        class Pipeline:
            def __init__(self):
                pass

            def __call__(self, input_: str):
                return {"some": "json serializable"}

        def get_pipeline():
            return Pipeline()

        routes = [
            Route("/{whatever:path}", status_ok),
            Route("/{whatever:path}", pipeline_route, methods=["POST"]),
        ]

        app = Starlette(routes=routes)

        @app.on_event("startup")
        async def startup_event():
            logger = logging.getLogger("uvicorn.access")
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.handlers = [handler]

            # Link between `api-inference-community` and framework code.
            app.get_pipeline = get_pipeline

        with TestClient(app) as client:
            response = client.post("/", data=b"")

        self.assertEqual(
            response.status_code,
            400,
        )
        self.assertEqual(
            response.content,
            b'{"error":"The task `invalid` is not recognized by api-inference-community"}',
        )

    def test_invalid_task(self):
        os.environ["TASK"] = "invalid"

        def get_pipeline():
            raise Exception("We cannot load the pipeline")

        routes = [
            Route("/{whatever:path}", status_ok),
            Route("/{whatever:path}", pipeline_route, methods=["POST"]),
        ]

        app = Starlette(routes=routes)

        @app.on_event("startup")
        async def startup_event():
            logger = logging.getLogger("uvicorn.access")
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.handlers = [handler]

            # Link between `api-inference-community` and framework code.
            app.get_pipeline = get_pipeline

        with TestClient(app) as client:
            response = client.post("/", data=b"")
        self.assertEqual(
            response.status_code,
            500,
        )
        self.assertEqual(response.content, b'{"error":"We cannot load the pipeline"}')

    def test_tts_pipeline(self):
        os.environ["TASK"] = "text-to-speech"

        class Pipeline:
            def __init__(self):
                pass

            def __call__(self, input_: str):
                return np.array([0, 0, 0]), 16000

        def get_pipeline():
            return Pipeline()

        routes = [
            Route("/{whatever:path}", status_ok),
            Route("/{whatever:path}", pipeline_route, methods=["POST"]),
        ]

        app = Starlette(routes=routes)

        @app.on_event("startup")
        async def startup_event():
            logger = logging.getLogger("uvicorn.access")
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.handlers = [handler]

            # Link between `api-inference-community` and framework code.
            app.get_pipeline = get_pipeline

        with TestClient(app) as client:
            response = client.post("/", data=b"2222")

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(response.headers["content-type"], "audio/flac")

    def test_audio_to_audio_pipeline(self):
        os.environ["TASK"] = "audio-to-audio"

        class Pipeline:
            def __init__(self):
                self.sampling_rate = 16000

            def __call__(self, input_: str):
                return np.array([[0, 0, 0]]), 16000, ["label_0"]

        def get_pipeline():
            return Pipeline()

        routes = [
            Route("/{whatever:path}", status_ok),
            Route("/{whatever:path}", pipeline_route, methods=["POST"]),
        ]

        app = Starlette(routes=routes)

        @app.on_event("startup")
        async def startup_event():
            logger = logging.getLogger("uvicorn.access")
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.handlers = [handler]

            # Link between `api-inference-community` and framework code.
            app.get_pipeline = get_pipeline

        bpayload = self.read("sample1.flac")
        with TestClient(app) as client:
            response = client.post("/", data=bpayload)

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertEqual(response.headers["x-compute-audio-length"], "13.69")
        data = json.loads(response.content)
        self.assertEqual(len(data), 1)
        self.assertEqual(set(data[0].keys()), {"blob", "label", "content-type"})
        self.assertEqual(data[0]["content-type"], "audio/flac")
        self.assertEqual(data[0]["label"], "label_0")

    def test_text_to_image_pipeline(self):
        os.environ["TASK"] = "text-to-image"

        class Pipeline:
            def __init__(self):
                pass

            def __call__(self, input_: str):
                dirname = os.path.dirname(os.path.abspath(__file__))
                filename = os.path.join(dirname, "samples", "plane.jpg")
                returned_image = Image.open(filename)
                return returned_image

        def get_pipeline():
            return Pipeline()

        routes = [
            Route("/{whatever:path}", status_ok),
            Route("/{whatever:path}", pipeline_route, methods=["POST"]),
        ]

        app = Starlette(routes=routes)

        @app.on_event("startup")
        async def startup_event():
            logger = logging.getLogger("uvicorn.access")
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.handlers = [handler]

            # Link between `api-inference-community` and framework code.
            app.get_pipeline = get_pipeline

        with TestClient(app) as client:
            response = client.post("/", data=b"")

        buf = io.BytesIO(response.content)
        image = Image.open(buf)
        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertTrue(isinstance(image, Image.Image))

    def test_pipeline_zero_shot(self):
        os.environ["TASK"] = "text-classification"

        class Pipeline:
            def __init__(self):
                pass

            def __call__(self, input_: str, candidate_labels=None):
                return {
                    "some": "json serializable",
                    "candidate_labels": candidate_labels,
                }

        def get_pipeline():
            return Pipeline()

        routes = [
            Route("/{whatever:path}", status_ok),
            Route("/{whatever:path}", pipeline_route, methods=["POST"]),
        ]

        app = Starlette(routes=routes)

        @app.on_event("startup")
        async def startup_event():
            logger = logging.getLogger("uvicorn.access")
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.handlers = [handler]

            # Link between `api-inference-community` and framework code.
            app.get_pipeline = get_pipeline

        with TestClient(app) as client:
            response = client.post(
                "/",
                json={"inputs": "Some", "parameters": {"candidate_labels": ["a", "b"]}},
            )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(response.headers["x-compute-characters"], "4")
        self.assertEqual(
            response.content,
            b'{"some":"json serializable","candidate_labels":["a","b"]}',
        )

    def test_image_classification_pipeline(self):
        os.environ["TASK"] = "image-classification"

        class Pipeline:
            def __init__(self):
                pass

            def __call__(self, input_: Image.Image):
                return [{"label_0": 1.0}]

        def get_pipeline():
            return Pipeline()

        routes = [
            Route("/{whatever:path}", status_ok),
            Route("/{whatever:path}", pipeline_route, methods=["POST"]),
        ]

        app = Starlette(routes=routes)

        @app.on_event("startup")
        async def startup_event():
            logger = logging.getLogger("uvicorn.access")
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.handlers = [handler]

            # Link between `api-inference-community` and framework code.
            app.get_pipeline = get_pipeline

        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, "samples", "plane.jpg")
        with open(filename, "rb") as f:
            data = f.read()
        with TestClient(app) as client:
            response = client.post("/", data=data)

        resp_data = json.loads(response.content)
        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(
            response.headers["x-compute-images"],
            "1",
        )
        self.assertEqual(resp_data, [{"label_0": 1.0}])
