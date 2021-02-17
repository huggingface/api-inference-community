import os
import sys
import unittest
from mimetypes import guess_type


SRC_DIR = os.path.join(os.path.dirname(__file__), "..")  # isort:skip
sys.path.append(SRC_DIR)  # isort:skip


import requests
from main import (
    EXAMPLE_ASR_EN_MODEL_ID,
    EXAMPLE_TTS_EN_MODEL_ID,
    EXAMPLE_TTS_ZH_MODEL_ID,
    HF_HEADER_COMPUTE_TIME,
    WAV2VEV2_MODEL_IDS,
)


ENDPOINT = "http://localhost:8000"


AUDIO_SAMPLE_LOCAL_FILES = [
    "samples/sample02-orig.wav",
    "samples/sample1.flac",
    "samples/chrome.webm",
    "samples/firefox.oga",
]

AUDIO_SAMPLE_REMOTE_URLS = [
    "https://cdn-media.huggingface.co/speech_samples/sample02-orig.wav",
    "https://cdn-media.huggingface.co/speech_samples/sample1.flac",
    "https://cdn-media.huggingface.co/speech_samples/chrome.webm",
    "https://cdn-media.huggingface.co/speech_samples/firefox.oga",
]


def endpoint_model(id: str) -> str:
    return f"{ENDPOINT}/models/{id}"


class AuxiliaryEndpointsTest(unittest.TestCase):
    def test_home(self):
        r = requests.get(ENDPOINT)
        r.raise_for_status()
        self.assertTrue(r.ok)

    def test_list_models(self):
        r = requests.get(f"{ENDPOINT}/models")
        r.raise_for_status()
        models = r.json()
        self.assertIsInstance(models, dict)


class TTSTest(unittest.TestCase):
    def test_tts_en(self):
        r = requests.post(
            url=endpoint_model(EXAMPLE_TTS_EN_MODEL_ID),
            json={"text": "My name is Julien"},
        )
        r.raise_for_status()
        print(r.headers.get(HF_HEADER_COMPUTE_TIME))
        self.assertEqual(r.headers.get("content-type"), "audio/x-wav")

    def test_tts_zh(self):
        r = requests.post(
            url=endpoint_model(EXAMPLE_TTS_ZH_MODEL_ID),
            json={"text": "请您说得慢些好吗"},
        )
        r.raise_for_status()
        print(r.headers.get(HF_HEADER_COMPUTE_TIME))
        self.assertEqual(r.headers.get("content-type"), "audio/x-wav")


class TimmTest(unittest.TestCase):
    def test_resnet50d_file_upload(self):
        with open(os.path.join(SRC_DIR, "samples/plane.jpg"), "rb") as f:
            r = requests.post(
                url=endpoint_model("sgugger/resnet50d"),
                data=f,
                headers={"content-type": "image/jpeg"},
            )
        r.raise_for_status()
        print(r.headers.get(HF_HEADER_COMPUTE_TIME))
        body = r.json()
        self.assertIsInstance(body, list)

    def test_resnet50d_url(self):
        r = requests.post(
            url=endpoint_model("sgugger/resnet50d"),
            json={"url": "https://huggingface.co/front/assets/transformers-demo.png"},
        )
        r.raise_for_status()
        print(r.headers.get(HF_HEADER_COMPUTE_TIME))
        body = r.json()
        self.assertIsInstance(body, list)


class ASRTest(unittest.TestCase):
    def test_asr_file_upload(self):
        # EXAMPLE_ASR_EN_MODEL_ID is too slow,
        # let's handle it separately if needed.
        for model_id in WAV2VEV2_MODEL_IDS:
            for audio_file in AUDIO_SAMPLE_LOCAL_FILES:
                with self.subTest():
                    with open(os.path.join(SRC_DIR, audio_file), "rb") as f:
                        r = requests.post(
                            url=endpoint_model(model_id),
                            data=f,
                            headers={"content-type": guess_type(audio_file)[0]},
                        )
                    r.raise_for_status()
                    print(r.headers.get(HF_HEADER_COMPUTE_TIME))
                    body = r.json()
                    print(body)
                    self.assertIsInstance(body, dict)

    def test_asr_url(self):
        # EXAMPLE_ASR_EN_MODEL_ID is too slow,
        # let's handle it separately if needed.
        for model_id in WAV2VEV2_MODEL_IDS:
            for audio_url in AUDIO_SAMPLE_REMOTE_URLS:
                with self.subTest():
                    r = requests.post(
                        url=endpoint_model(model_id),
                        json={"url": audio_url},
                    )
                    r.raise_for_status()
                    print(r.headers.get(HF_HEADER_COMPUTE_TIME))
                    body = r.json()
                    print(body)
                    self.assertIsInstance(body, dict)
