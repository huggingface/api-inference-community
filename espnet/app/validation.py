import json
import subprocess
from io import BytesIO
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urlparse

import httpx
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConstrainedFloat, ConstrainedInt, ConstrainedList


class MaxLength(ConstrainedInt):
    ge = 1
    le = 500
    strict = True


class TopK(ConstrainedInt):
    ge = 1
    strict = True


class TopP(ConstrainedFloat):
    ge = 0.0
    le = 1.0
    strict = True


class RepetitionPenalty(ConstrainedFloat):
    ge = 0.0
    le = 100.0
    strict = True


class Temperature(ConstrainedFloat):
    ge = 0.0
    le = 100.0
    strict = True


class TextGenerationCheck(BaseModel):
    max_length: Optional[MaxLength] = None
    top_k: Optional[TopK] = None
    top_p: Optional[TopP] = None
    repetition_penalty: Optional[RepetitionPenalty] = None
    temperature: Optional[Temperature] = None


class FillMaskCheck(BaseModel):
    top_k: Optional[TopK] = None


class CandidateLabels(ConstrainedList):
    min_items = 1
    __args__ = [str]


class ZeroShotCheck(BaseModel):
    candidate_labels: Union[str, CandidateLabels]
    multi_class: Optional[bool] = None
    multi_label: Optional[bool] = None


MAPPING = {
    "conversational": TextGenerationCheck,
    "text-generation": TextGenerationCheck,
    "fill-mask": FillMaskCheck,
    "zero-shot-classification": ZeroShotCheck,
}


def check_params(params, tag):
    if tag in MAPPING:
        MAPPING[tag](**params)
    return True


def normalize_payload(bpayload: bytes, task: str) -> Tuple[Any, Dict]:
    if task in {
        "automatic-speech-recognition",
    }:
        return normalize_payload_audio(bpayload)
    elif task in {
        "image-classification",
    }:
        return normalize_payload_image(bpayload)
    else:
        return normalize_payload_nlp(bpayload, task)


def ffmpeg_convert(array: np.array, sampling_rate: int) -> bytes:
    """
    Helper function to convert raw waveforms to actual compressed file (lossless compression here)
    """
    ar = str(sampling_rate)
    ac = "1"
    format_for_conversion = "flac"
    ffmpeg_command = [
        "ffmpeg",
        "-ac",
        "1",
        "-f",
        "f32le",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    ffmpeg_process = subprocess.Popen(
        ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    output_stream = ffmpeg_process.communicate(array.tobytes())
    out_bytes = output_stream[0]
    if len(out_bytes) == 0:
        raise Exception("Impossible to convert output stream")
    return out_bytes


def ffmpeg_read(bpayload: bytes) -> np.array:
    """
    Librosa does that under the hood but forces the use of an actual
    file leading to hitting disk, which is almost always very bad.
    """
    ar = "16k"
    ac = "1"
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    ffmpeg_process = subprocess.Popen(
        ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    output_stream = ffmpeg_process.communicate(bpayload)
    out_bytes = output_stream[0]

    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError("Malformed soundfile")
    return audio


def normalize_payload_image(bpayload: bytes) -> Tuple[Any, Dict]:
    img = Image.open(BytesIO(bpayload))
    return img, {}


def normalize_payload_audio(bpayload: bytes) -> Tuple[Any, Dict]:
    exc = None
    try:
        data = json.loads(bpayload)
        if "url" in data:
            parsed = urlparse(data["url"])
            if parsed.netloc != "cdn-media.huggingface.co":
                exc = ValueError(
                    "We don't support any other domain than `cdn-media.huggingface.co"
                )
                raise Exception("Break")
            bpayload = httpx.get(data["url"], timeout=2).content
    except Exception:
        pass
    if exc is not None:
        raise exc
    inputs = ffmpeg_read(bpayload)
    if len(inputs.shape) > 1:
        # ogg can take dual channel input -> take only first input channel in this case
        inputs = inputs[:, 0]
    return inputs, {}


def normalize_payload_nlp(bpayload: bytes, task: str) -> Tuple[Any, Dict]:
    payload = bpayload.decode("utf-8")

    # We used to accept raw strings, we need to maintain backward compatibility
    try:
        payload = json.loads(payload)
    except Exception:
        pass

    parameters: Dict[str, Any] = {}
    if isinstance(payload, dict) and "inputs" in payload:
        inputs = payload["inputs"]
        parameters = payload.get("parameters", {})
    else:
        inputs = payload
    check_params(parameters, task)
    return inputs, parameters
