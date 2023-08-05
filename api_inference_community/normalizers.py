"""
Helper classes to modify pipeline outputs from tensors to expected pipeline output
"""

import json
import os
from base64 import b64decode
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from api_inference_community.constants import (
    AUDIO,
    AUDIO_INPUTS,
    IMAGE_INPUTS,
    TEXT_INPUTS,
)
from api_inference_community.validation import check_inputs, check_params, ffmpeg_read


Classes = Dict[str, Union[str, float]]
DATA_PREFIX = os.getenv("HF_TRANSFORMERS_CACHE", "")

if TYPE_CHECKING:
    try:
        import torch
    except Exception:
        pass


def speaker_diarization_normalize(
    tensor: "torch.Tensor", sampling_rate: int, classnames: List[str]
) -> List[Classes]:
    N = tensor.shape[1]
    if len(classnames) != N:
        raise ValueError(
            f"There is a mismatch between classnames ({len(classnames)}) and number of speakers ({N})"
        )
    classes = []
    for i in range(N):
        values, counts = tensor[:, i].unique_consecutive(return_counts=True)
        offset = 0
        for v, c in zip(values, counts):
            if v == 1:
                classes.append(
                    {
                        "class": classnames[i],
                        "start": offset / sampling_rate,
                        "end": (offset + c.item()) / sampling_rate,
                    }
                )
            offset += c.item()

    classes = sorted(classes, key=lambda x: x["start"])
    return classes


def normalize_payload(
    bpayload: bytes, task: str, sampling_rate: Optional[int]
) -> Tuple[Any, Dict]:
    if task in AUDIO_INPUTS:
        if sampling_rate is None:
            raise EnvironmentError(
                "We cannot normalize audio file if we don't know the sampling rate"
            )
        return normalize_payload_audio(bpayload, sampling_rate)
    elif task in IMAGE_INPUTS:
        return normalize_payload_image(bpayload)
    elif task in TEXT_INPUTS:
        return normalize_payload_nlp(bpayload, task)
    else:
        raise EnvironmentError(
            f"The task `{task}` is not recognized by api-inference-community"
        )


def normalize_payload_image(bpayload: bytes) -> Tuple[Any, Dict]:
    from PIL import Image

    try:
        # We accept both binary image with mimetype
        # and {"inputs": base64encodedimage}
        data = json.loads(bpayload)
        image = data["image"] if "image" in data else data["inputs"]
        image_bytes = b64decode(image)
        img = Image.open(BytesIO(image_bytes))
        return img, data.get("parameters", {})
    except Exception:
        pass

    img = Image.open(BytesIO(bpayload))
    return img, {}


def normalize_payload_audio(bpayload: bytes, sampling_rate: int) -> Tuple[Any, Dict]:
    if os.path.isfile(bpayload) and bpayload.startswith(DATA_PREFIX.encode("utf-8")):
        # XXX:
        # This is necessary for batch jobs where the datasets can contain
        # filenames instead of the raw data.
        # We attempt to sanitize this roughly, by checking it lives on the data
        # path (hardcoded in the deployment and in all the dockerfiles)
        # We also attempt to prevent opening files that are not obviously
        # audio files, to prevent opening stuff like model weights.
        filename, ext = os.path.splitext(bpayload)
        if ext.decode("utf-8")[1:] in AUDIO:
            with open(bpayload, "rb") as f:
                bpayload = f.read()
    inputs = ffmpeg_read(bpayload, sampling_rate)
    if len(inputs.shape) > 1:
        # ogg can take dual channel input -> take only first input channel in this case
        inputs = inputs[:, 0]
    return inputs, {}


def normalize_payload_nlp(bpayload: bytes, task: str) -> Tuple[Any, Dict]:
    payload = bpayload.decode("utf-8")

    # We used to accept raw strings, we need to maintain backward compatibility
    try:
        payload = json.loads(payload)
        if isinstance(payload, (float, int)):
            payload = str(payload)
    except Exception:
        pass

    parameters: Dict[str, Any] = {}
    if isinstance(payload, dict) and "inputs" in payload:
        inputs = payload["inputs"]
        parameters = payload.get("parameters", {})
    else:
        inputs = payload
    check_params(parameters, task)
    check_inputs(inputs, task)
    return inputs, parameters
