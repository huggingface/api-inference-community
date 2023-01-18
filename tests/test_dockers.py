import json
import os
import subprocess
import time
import unittest
import uuid
from collections import Counter
from typing import Any, Optional

import httpx


class DockerPopen(subprocess.Popen):
    def __exit__(self, exc_type, exc_val, traceback):
        self.terminate()
        self.wait(20)
        return super().__exit__(exc_type, exc_val, traceback)


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


@unittest.skipIf(
    "RUN_DOCKER_TESTS" not in os.environ,
    "Docker tests are slow, set `RUN_DOCKER_TESTS=1` environement variable to run them",
)
class DockerImageTests(unittest.TestCase):
    def create_docker(self, name: str) -> str:
        rand = str(uuid.uuid4())[:5]
        tag = f"{name}:{rand}"
        with cd(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "docker_images", name
            )
        ):
            proc = subprocess.run(["docker", "build", ".", "-t", tag])
        self.assertEqual(proc.returncode, 0)
        return tag

    def test_allennlp(self):
        self.framework_docker_test(
            "allennlp", "question-answering", "lysandre/bidaf-elmo-model-2020.03.19"
        )
        self.framework_invalid_test("allennlp")

    def test_asteroid(self):
        self.framework_docker_test(
            "asteroid",
            "audio-to-audio",
            "mhu-coder/ConvTasNet_Libri1Mix_enhsingle",
        )
        self.framework_docker_test(
            "asteroid",
            "audio-to-audio",
            "julien-c/DPRNNTasNet-ks16_WHAM_sepclean",
        )
        self.framework_invalid_test("asteroid")

    def test_espnet(self):
        self.framework_docker_test(
            "espnet",
            "text-to-speech",
            "espnet/kan-bayashi_ljspeech_fastspeech2",
        )
        self.framework_invalid_test("espnet")
        self.framework_docker_test(
            "espnet",
            "automatic-speech-recognition",
            "espnet/kamo-naoyuki_mini_an4_asr_train_raw_bpe_valid.acc.best",
        )

    def test_fairseq(self):
        self.framework_docker_test(
            "fairseq",
            "text-to-speech",
            "facebook/fastspeech2-en-ljspeech",
        )
        self.framework_docker_test(
            "fairseq",
            "audio-to-audio",
            "facebook/xm_transformer_600m-es_en-multi_domain",
        )
        self.framework_invalid_test("fairseq")

    def test_fasttext(self):
        self.framework_docker_test(
            "fasttext",
            "text-classification",
            "osanseviero/fasttext_nearest",
        )
        self.framework_docker_test(
            "fasttext",
            "feature-extraction",
            "osanseviero/fasttext_embedding",
        )
        self.framework_invalid_test("fasttext")

    def test_sentence_transformers(self):
        self.framework_docker_test(
            "sentence_transformers",
            "feature-extraction",
            "bert-base-uncased",
        )

        self.framework_docker_test(
            "sentence_transformers",
            "sentence-similarity",
            "sentence-transformers/paraphrase-distilroberta-base-v1",
        )
        self.framework_invalid_test("sentence_transformers")

    def test_adapter_transformers(self):
        self.framework_docker_test(
            "adapter_transformers",
            "question-answering",
            "calpt/adapter-bert-base-squad1",
        )

        self.framework_docker_test(
            "adapter_transformers",
            "text-classification",
            "AdapterHub/roberta-base-pf-sick",
        )

        self.framework_docker_test(
            "adapter_transformers",
            "token-classification",
            "AdapterHub/roberta-base-pf-conll2003",
        )

        self.framework_invalid_test("adapter_transformers")

    def test_flair(self):
        self.framework_docker_test(
            "flair", "token-classification", "flair/chunk-english-fast"
        )
        self.framework_invalid_test("flair")

    def test_paddlenlp(self):
        self.framework_docker_test(
            "paddlenlp", "fill-mask", "PaddleCI/tiny-random-ernie"
        )
        self.framework_docker_test(
            "paddlenlp", "conversational", "PaddleCI/tiny-random-plato-mini"
        )
        self.framework_docker_test(
            "paddlenlp", "zero-shot-classification", "PaddleCI/tiny-random-ernie"
        )
        self.framework_docker_test(
            "paddlenlp", "summarization", "PaddleCI/tiny-random-unimo-text-1.0"
        )
        self.framework_invalid_test("paddlenlp")

    def test_sklearn(self):
        clf_data = {
            "data": {
                "sepal length (cm)": [6.1, 5.7, 7.7],
                "sepal width (cm)": [2.8, 3.8, 2.6],
                "petal length (cm)": [4.7, 1.7, 6.9],
                "petal width (cm)": [1.2, 0.3, 2.3],
            }
        }
        self.framework_docker_test(
            "sklearn",
            "tabular-classification",
            "skops-tests/iris-sklearn-latest-logistic_regression-with-config",
            custom_input=clf_data,
            timeout=600,
        )

        regr_data = {
            "data": {
                "age": [0.045, 0.092, 0.063],
                "sex": [-0.044, -0.044, 0.050],
                "bmi": [-0.006, 0.036, -0.004],
                "bp": [-0.015, 0.021, -0.012],
                "s1": [0.125, -0.024, 0.103],
                "s2": [0.125, -0.016, 0.048],
                "s3": [0.019, 0.000, 0.056],
                "s4": [0.034, -0.039, -0.002],
                "s5": [0.032, -0.022, 0.084],
                "s6": [-0.005, -0.021, -0.017],
            }
        }
        self.framework_docker_test(
            "sklearn",
            "tabular-regression",
            "skops-tests/tabularregression-sklearn-latest-linear_regression-with-config",
            custom_input=regr_data,
            timeout=600,
        )
        self.framework_docker_test(
            "sklearn",
            "text-classification",
            "merve/20newsgroups",
            timeout=600,
        )

    def test_k2(self):
        self.framework_docker_test(
            "k2",
            "automatic-speech-recognition",
            "csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13",
        )

    def test_spacy(self):
        self.framework_docker_test(
            "spacy",
            "token-classification",
            "spacy/en_core_web_sm",
        )
        self.framework_docker_test(
            "spacy",
            "text-classification",
            "cverluise/xx_cat_pateexx_md",
        )
        self.framework_docker_test(
            "spacy",
            "sentence-similarity",
            "spacy/en_core_web_sm",
        )
        self.framework_invalid_test("spacy")

    def test_speechbrain(self):
        self.framework_docker_test(
            "speechbrain",
            "automatic-speech-recognition",
            "speechbrain/asr-crdnn-commonvoice-it",
        )

        self.framework_docker_test(
            "speechbrain",
            "automatic-speech-recognition",
            "speechbrain/asr-wav2vec2-commonvoice-fr",
        )

        self.framework_docker_test(
            "speechbrain",
            "text-to-speech",
            "speechbrain/tts-tacotron2-ljspeech",
        )

        self.framework_invalid_test("speechbrain")

        # source-separation
        self.framework_docker_test(
            "speechbrain",
            "audio-to-audio",
            "speechbrain/sepformer-wham",
        )

        # speech-enchancement
        self.framework_docker_test(
            "speechbrain",
            "audio-to-audio",
            "speechbrain/mtl-mimic-voicebank",
        )

        self.framework_docker_test(
            "speechbrain",
            "audio-classification",
            "speechbrain/urbansound8k_ecapa",
        )

    def test_stanza(self):
        self.framework_docker_test(
            "stanza", "token-classification", "stanfordnlp/stanza-en"
        )

        self.framework_docker_test(
            "stanza",
            "token-classification",
            "stanfordnlp/stanza-tr",
        )
        self.framework_invalid_test("stanza")

    def test_timm(self):
        self.framework_docker_test("timm", "image-classification", "sgugger/resnet50d")
        self.framework_invalid_test("timm")

    def test_diffusers(self):
        self.framework_docker_test(
            "diffusers",
            "text-to-image",
            "hf-internal-testing/tiny-stable-diffusion-pipe",
        )
        self.framework_invalid_test("diffusers")

    def test_pyannote_audio(self):
        self.framework_docker_test(
            "pyannote_audio",
            "automatic-speech-recognition",
            "pyannote/voice-activity-detection",
        )
        self.framework_invalid_test("pyannote_audio")

    def test_keras(self):
        # Single Output Unit, RGB
        self.framework_docker_test(
            "keras", "image-classification", "nateraw/keras-cats-vs-dogs"
        )
        # Multiple Output Units, Grayscale
        self.framework_docker_test(
            "keras", "image-classification", "nateraw/keras-mnist-convnet"
        )

    def test_fastai(self):
        # Single Output Unit, RGB
        self.framework_docker_test(
            "fastai", "image-classification", "fastai/fastbook_02_bears_classifier"
        )
        self.framework_docker_test(
            "fastai",
            "image-classification",
            "Kieranm/britishmus_plate_material_classifier",
        )
        self.framework_invalid_test("fastai")

    def test_doctr(self):
        self.framework_docker_test(
            "doctr", "object-detection", "mindee/fasterrcnn_mobilenet_v3_large_fpn"
        )
        self.framework_invalid_test("doctr")

    def test_nemo(self):
        self.framework_docker_test(
            "nemo", "automatic-speech-recognition", "nvidia/stt_en_conformer_ctc_medium"
        )

    def test_generic(self):
        self.framework_docker_test(
            "generic",
            "token-classification",
            "osanseviero/en_core_web_sm",
        )

    def framework_invalid_test(self, framework: str):
        task = "invalid"
        model_id = "invalid"
        tag = self.create_docker(framework)
        run_docker_command = [
            "docker",
            "run",
            "-p",
            "8000:80",
            "-e",
            f"TASK={task}",
            "-e",
            f"MODEL_ID={model_id}",
            "-v",
            "/tmp:/data",
            "-t",
            tag,
        ]

        url = "http://localhost:8000"
        timeout = 60
        with DockerPopen(run_docker_command) as proc:
            for i in range(400):
                try:
                    response = httpx.get(url, timeout=10)
                    break
                except Exception:
                    time.sleep(1)
            self.assertEqual(response.content, b'{"ok":"ok"}')

            response = httpx.post(url, data=b"This is a test", timeout=timeout)
            self.assertIn(response.status_code, {400, 500})
            self.assertEqual(response.headers["content-type"], "application/json")

            proc.terminate()
            proc.wait(20)

    def framework_docker_batch(
        self,
        framework: str,
        task: str,
        model_id: str,
        dataset_name: str,
        dataset_config: str,
        dataset_split: str,
        dataset_column: str,
    ):
        tag = self.create_docker(framework)
        run_docker_command = [
            "docker",
            "run",
            "-p",
            "8000:80",
            "-it",
            "-e",
            f"TASK={task}",
            "-e",
            f"MODEL_ID={model_id}",
            "-e",
            f"DATASET_NAME={dataset_name}",
            "-e",
            f"DATASET_CONFIG={dataset_config}",
            "-e",
            f"DATASET_SPLIT={dataset_split}",
            "-e",
            f"DATASET_COLUMN={dataset_column}",
            "-v",
            "/tmp:/data",
            "-t",
            tag,
            "python",
            "app/batch.py",
        ]

        with DockerPopen(run_docker_command) as proc:
            proc.wait()
        self.assertTrue(True)

    def framework_docker_test(
        self,
        framework: str,
        task: str,
        model_id: str,
        custom_input: Optional[
            Any
        ] = None,  # if given, check inference with this specific input
        timeout=60,
    ):
        tag = self.create_docker(framework)
        run_docker_command = [
            "docker",
            "run",
            "-p",
            "8000:80",
            "-e",
            f"TASK={task}",
            "-e",
            f"MODEL_ID={model_id}",
            "-v",
            "/tmp:/data",
            "-t",
            tag,
        ]

        url = "http://localhost:8000"
        counter = Counter()
        with DockerPopen(run_docker_command) as proc:
            for i in range(400):
                try:
                    response = httpx.get(url, timeout=10)
                    break
                except Exception:
                    time.sleep(1)
            self.assertEqual(response.content, b'{"ok":"ok"}')

            response = httpx.post(url, data=b"This is a test", timeout=timeout)
            self.assertIn(response.status_code, {200, 400}, response.content)
            counter[response.status_code] += 1

            response = httpx.post(
                url,
                # Include the mask for fill-mask tests.
                json={"inputs": "This is a test [MASK]"},
                timeout=timeout,
            )
            self.assertIn(response.status_code, {200, 400})
            counter[response.status_code] += 1

            response = httpx.post(
                url,
                # Conversational
                json={
                    "inputs": {
                        "text": "My name [MASK].",
                        "past_user_inputs": [],
                        "generated_responses": [],
                    }
                },
                timeout=timeout,
            )
            self.assertIn(response.status_code, {200, 400})
            counter[response.status_code] += 1

            response = httpx.post(
                url,
                json={
                    "inputs": {"question": "This is a test", "context": "Some context"}
                },
                timeout=timeout,
            )
            self.assertIn(response.status_code, {200, 400})
            counter[response.status_code] += 1

            response = httpx.post(
                url,
                json={
                    "inputs": {
                        "source_sentence": "This is a test",
                        "sentences": ["Some context", "Something else"],
                    }
                },
                timeout=timeout,
            )
            self.assertIn(response.status_code, {200, 400}, response.content)
            counter[response.status_code] += 1

            response = httpx.post(
                url,
                json={
                    "inputs": "This is a test",
                    "parameters": {"candidate_labels": ["a", "b"]},
                },
                timeout=timeout,
            )
            self.assertIn(response.status_code, {200, 400}, response.content)
            counter[response.status_code] += 1

            response = httpx.post(
                url,
                json={
                    "inputs": {
                        "data": {
                            "1": [7.4],
                            "2": [7.5],
                            "3": [7.7],
                            "4": [7.7],
                            "5": [7.7],
                            "6": [7.7],
                            "7": [7.7],
                            "8": [7.7],
                            "9": [7.7],
                            "10": [7.7],
                            "11": [7.7],
                        }
                    }
                },
                timeout=timeout,
            )
            self.assertIn(response.status_code, {200, 400})
            counter[response.status_code] += 1

            if custom_input is not None:
                response = httpx.post(
                    url,
                    json=custom_input,
                    timeout=timeout,
                )
                self.assertIn(response.status_code, {200, 400})
                counter[response.status_code] += 1

            with open(
                os.path.join(os.path.dirname(__file__), "samples", "sample1.flac"), "rb"
            ) as f:
                data = f.read()
            response = httpx.post(url, data=data, timeout=timeout)
            self.assertIn(response.status_code, {200, 400})
            counter[response.status_code] += 1
            if response.status_code == 200:
                if response.headers["content-type"] == "application/json":
                    data = json.loads(response.content)
                    if isinstance(data, dict):
                        # ASR
                        self.assertEqual(set(data.keys()), {"text"})
                    elif isinstance(data, list):
                        if len(data) > 0:
                            keys = set(data[0].keys())
                            if keys == {"blob", "content-type", "label"}:
                                # audio-to-audio
                                self.assertEqual(
                                    keys, {"blob", "content-type", "label"}
                                )
                            else:
                                speech_segmentation_keys = {"class", "start", "end"}
                                audio_classification_keys = {"label", "score"}
                                self.assertIn(
                                    keys,
                                    [
                                        audio_classification_keys,
                                        speech_segmentation_keys,
                                    ],
                                )
                    else:
                        raise Exception("Invalid result")
                elif response.headers["content-type"] == "audio/flac":
                    pass
                else:
                    raise Exception("Unknown format")

            with open(
                os.path.join(os.path.dirname(__file__), "samples", "malformed.flac"),
                "rb",
            ) as f:
                data = f.read()
            response = httpx.post(url, data=data, timeout=timeout)
            self.assertIn(response.status_code, {200, 400}, response.content)
            counter[response.status_code] += 1

            with open(
                os.path.join(os.path.dirname(__file__), "samples", "plane.jpg"), "rb"
            ) as f:
                data = f.read()
            response = httpx.post(url, data=data, timeout=timeout)
            self.assertIn(response.status_code, {200, 400})
            counter[response.status_code] += 1

            with open(
                os.path.join(os.path.dirname(__file__), "samples", "sample1_dual.ogg"),
                "rb",
            ) as f:
                data = f.read()
            response = httpx.post(url, data=data, timeout=timeout)
            self.assertIn(response.status_code, {200, 400})
            counter[response.status_code] += 1

            with open(
                os.path.join(os.path.dirname(__file__), "samples", "sample1.webm"), "rb"
            ) as f:
                data = f.read()
            response = httpx.post(url, data=data, timeout=timeout)
            self.assertIn(response.status_code, {200, 400})
            counter[response.status_code] += 1

            proc.terminate()
            proc.wait(20)

        self.assertEqual(proc.returncode, 0)
        self.assertGreater(
            counter[200],
            0,
            f"At least one request should have gone through {framework}, {task}, {model_id}",
        )

        # Follow up loading are much faster, 20s should be ok.
        with DockerPopen(run_docker_command) as proc2:
            for i in range(20):
                try:
                    response2 = httpx.get(url, timeout=10)
                    break
                except Exception:
                    time.sleep(1)
            self.assertEqual(response2.content, b'{"ok":"ok"}')
            proc2.terminate()
            proc2.wait(20)
        self.assertEqual(proc2.returncode, 0)
