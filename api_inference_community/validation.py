import json
import os
import subprocess
from base64 import b64decode
from io import BytesIO
from mimetypes import MimeTypes
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import (
    BaseModel,
    ConstrainedFloat,
    ConstrainedInt,
    ConstrainedList,
    validator,
)


class MinLength(ConstrainedInt):
    ge = 1
    le = 500
    strict = True


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


class MaxTime(ConstrainedFloat):
    ge = 0.0
    le = 120.0
    strict = True


class NumReturnSequences(ConstrainedInt):
    ge = 1
    le = 10
    strict = True


class RepetitionPenalty(ConstrainedFloat):
    ge = 0.0
    le = 100.0
    strict = True


class Temperature(ConstrainedFloat):
    ge = 0.0
    le = 100.0
    strict = True


class CandidateLabels(ConstrainedList):
    min_items = 1
    __args__ = [str]


class FillMaskParamsCheck(BaseModel):
    top_k: Optional[TopK] = None


class ZeroShotParamsCheck(BaseModel):
    candidate_labels: Union[str, CandidateLabels]
    multi_label: Optional[bool] = None


class SharedGenerationParams(BaseModel):
    min_length: Optional[MinLength] = None
    max_length: Optional[MaxLength] = None
    top_k: Optional[TopK] = None
    top_p: Optional[TopP] = None
    max_time: Optional[MaxTime] = None
    repetition_penalty: Optional[RepetitionPenalty] = None
    temperature: Optional[Temperature] = None

    @validator("max_length")
    def max_length_must_be_larger_than_min_length(
        cls, max_length: Optional[MinLength], values: Dict[str, Optional[str]]
    ):
        if "min_length" in values:
            if values["min_length"] is not None:
                if max_length < values["min_length"]:
                    raise ValueError("min_length cannot be larger than max_length")
        return max_length


class TextGenerationParamsCheck(SharedGenerationParams):
    return_full_text: Optional[bool] = None
    num_return_sequences: Optional[NumReturnSequences] = None


class SummarizationParamsCheck(SharedGenerationParams):
    num_return_sequences: Optional[NumReturnSequences] = None


class ConversationalInputsCheck(BaseModel):
    text: str
    past_user_inputs: List[str]
    generated_responses: List[str]


class QuestionInputsCheck(BaseModel):
    question: str
    context: str


class SentenceSimilarityInputsCheck(BaseModel):
    source_sentence: str
    sentences: List[str]


class TableQuestionAnsweringInputsCheck(BaseModel):
    table: Dict[str, List[str]]
    query: str

    @validator("table")
    def all_rows_must_have_same_length(cls, table: Dict[str, List[str]]):
        rows = list(table.values())
        n = len(rows[0])
        if all(len(x) == n for x in rows):
            return table
        raise ValueError("All rows in the table must be the same length")


class TabularDataInputsCheck(BaseModel):
    data: Dict[str, List[str]]

    @validator("data")
    def all_rows_must_have_same_length(cls, data: Dict[str, List[str]]):
        rows = list(data.values())
        n = len(rows[0])
        if all(len(x) == n for x in rows):
            return data
        raise ValueError("All rows in the data must be the same length")


class StringOrStringBatchInputCheck(BaseModel):
    __root__: Union[List[str], str]

    @validator("__root__")
    def input_must_not_be_empty(cls, __root__: Union[List[str], str]):
        if isinstance(__root__, list):
            if len(__root__) == 0:
                raise ValueError(
                    "The inputs are invalid, at least one input is required"
                )
        return __root__


class StringInput(BaseModel):
    __root__: str


PARAMS_MAPPING = {
    "conversational": SharedGenerationParams,
    "fill-mask": FillMaskParamsCheck,
    "text2text-generation": TextGenerationParamsCheck,
    "text-generation": TextGenerationParamsCheck,
    "summarization": SummarizationParamsCheck,
    "zero-shot-classification": ZeroShotParamsCheck,
}


INPUTS_MAPPING = {
    "conversational": ConversationalInputsCheck,
    "question-answering": QuestionInputsCheck,
    "feature-extraction": StringOrStringBatchInputCheck,
    "sentence-similarity": SentenceSimilarityInputsCheck,
    "table-question-answering": TableQuestionAnsweringInputsCheck,
    "tabular-classification": TabularDataInputsCheck,
    "tabular-regression": TabularDataInputsCheck,
    "fill-mask": StringInput,
    "summarization": StringInput,
    "text2text-generation": StringInput,
    "text-generation": StringInput,
    "text-classification": StringInput,
    "token-classification": StringInput,
    "translation": StringInput,
    "zero-shot-classification": StringInput,
    "text-to-speech": StringInput,
    "text-to-image": StringInput,
}


BATCH_ENABLED_PIPELINES = ["feature-extraction"]


def check_params(params, tag):
    if tag in PARAMS_MAPPING:
        PARAMS_MAPPING[tag].parse_obj(params)
    return True


def check_inputs(inputs, tag):
    if tag in INPUTS_MAPPING:
        INPUTS_MAPPING[tag].parse_obj(inputs)
        return True
    else:
        raise ValueError(f"{tag} is not a valid pipeline.")


AUDIO_INPUTS = {
    "automatic-speech-recognition",
    "audio-to-audio",
    "speech-segmentation",
    "audio-classification",
}


IMAGE_INPUTS = {
    "image-classification",
    "image-segmentation",
    "image-to-text",
    "image-to-image",
    "object-detection",
    "zero-shot-image-classification",
}


TEXT_INPUTS = {
    "conversational",
    "feature-extraction",
    "question-answering",
    "sentence-similarity",
    "fill-mask",
    "table-question-answering",
    "tabular-classification",
    "tabular-regression",
    "summarization",
    "text2text-generation",
    "text-classification",
    "text-to-image",
    "text-to-speech",
    "token-classification",
    "zero-shot-classification",
}


WHITELISTED_MIME_TYPES = {
    "audio/flac": "flac",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/ogg": "ogg",
    "audio/mp4": "m4a",
    "audio/aac": "aac",
    "audio/webm": "webm",
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
    "image/webp": "webp",
}


def normalize_payload(
    bpayload: bytes, task: str, sampling_rate: Optional[int] = None, accept_header: Optional[str] = None
) -> Tuple[Any, Dict]:

    if accept_header:
        mime = MimeTypes()
        requested_formats = {mime_type: mime.guess_extension(mime_type).lstrip('.') for mime_type in accept_header.split(',')}
    else:
        requested_formats = {}

    if task in AUDIO_INPUTS:
        if sampling_rate is None:
            raise EnvironmentError(
                "We cannot normalize audio file if we don't know the sampling rate"
            )
        outputs = normalize_payload_audio(bpayload, sampling_rate)
        audio_format = "flac"

        for requested_format in requested_formats.values():
            if requested_format in WHITELISTED_MIME_TYPES.values():
                audio_format = requested_format
                break

        return outputs, audio_format
    elif task in IMAGE_INPUTS:
        outputs = normalize_payload_image(bpayload)
        image_format = "jpeg"

        for requested_format in requested_formats.values():
            if requested_format in WHITELISTED_MIME_TYPES.values():
                image_format = requested_format
                break

        return outputs, image_format
    elif task in TEXT_INPUTS:
        return normalize_payload_nlp(bpayload, task)
    else:
        raise EnvironmentError(
            f"The task `{task}` is not recognized by api-inference-community"
        )


def ffmpeg_convert(array: np.array, sampling_rate: int, format_for_conversion: str) -> bytes:
    """
    Helper function to convert raw waveforms to actual compressed file (lossless compression here)
    """
    ar = str(sampling_rate)
    ac = "1"
    ffmpeg_command = [
        "ffmpeg",
        "-ac",
        "1",
        "-f",
        "f32le",
        "-ac",
        ac,
        "-ar",
        ar,
        "-i",
        "pipe:0",
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


def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Librosa does that under the hood but forces the use of an actual
    file leading to hitting disk, which is almost always very bad.
    """
    ar = f"{sampling_rate}"
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

    audio = np.frombuffer(out_bytes, np.float32).copy()
    if audio.shape[0] == 0:
        raise ValueError("Malformed soundfile")
    return audio


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


DATA_PREFIX = os.getenv("HF_TRANSFORMERS_CACHE", "")


def normalize_payload_audio(bpayload: bytes, sampling_rate: int) -> Tuple[Any, Dict]:
    audio_extensions = {
        f".{ext}" for mime_type, ext in WHITELISTED_MIME_TYPES.items() if "audio" in mime_type
    }

    if os.path.isfile(bpayload) and bpayload.startswith(DATA_PREFIX.encode("utf-8")):
        # XXX:
        # This is necessary for batch jobs where the datasets can contain
        # filenames instead of the raw data.
        # We attempt to sanitize this roughly, by checking it lives on the data
        # path (hardcoded in the deployment and in all the dockerfiles)
        # We also attempt to prevent opening files that are not obviously
        # audio files, to prevent opening stuff like model weights.
        filename, ext = os.path.splitext(bpayload)
        if ext.decode("utf-8") in audio_extensions:
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
