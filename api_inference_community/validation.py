import json
import os
import subprocess
from base64 import b64decode
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import annotated_types
import numpy as np
from pydantic import BaseModel, RootModel, Strict, field_validator
from typing_extensions import Annotated


MinLength = Annotated[int, annotated_types.Ge(1), annotated_types.Le(500), Strict()]
MaxLength = Annotated[int, annotated_types.Ge(1), annotated_types.Le(500), Strict()]
TopK = Annotated[int, annotated_types.Ge(1), Strict()]
TopP = Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(1.0), Strict()]
MaxTime = Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(120.0), Strict()]
NumReturnSequences = Annotated[
    int, annotated_types.Ge(1), annotated_types.Le(10), Strict()
]
RepetitionPenalty = Annotated[
    float, annotated_types.Ge(0.0), annotated_types.Le(100.0), Strict()
]
Temperature = Annotated[
    float, annotated_types.Ge(0.0), annotated_types.Le(100.0), Strict()
]
CandidateLabels = Annotated[list, annotated_types.MinLen(1)]


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

    @field_validator("max_length")
    def max_length_must_be_larger_than_min_length(
        cls, max_length: Optional[MaxLength], values
    ):
        min_length = values.data.get("min_length", 0)
        if min_length is None:
            min_length = 0
        if max_length is not None and max_length < min_length:
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

    @field_validator("table")
    def all_rows_must_have_same_length(cls, table: Dict[str, List[str]]):
        rows = list(table.values())
        n = len(rows[0])
        if all(len(x) == n for x in rows):
            return table
        raise ValueError("All rows in the table must be the same length")


class TabularDataInputsCheck(BaseModel):
    data: Dict[str, List[str]]

    @field_validator("data")
    def all_rows_must_have_same_length(cls, data: Dict[str, List[str]]):
        rows = list(data.values())
        n = len(rows[0])
        if all(len(x) == n for x in rows):
            return data
        raise ValueError("All rows in the data must be the same length")


class StringOrStringBatchInputCheck(RootModel):
    root: Union[List[str], str]

    @field_validator("root")
    def input_must_not_be_empty(cls, root: Union[List[str], str]):
        if isinstance(root, list):
            if len(root) == 0:
                raise ValueError(
                    "The inputs are invalid, at least one input is required"
                )
        return root


class StringInput(RootModel):
    root: str


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
        PARAMS_MAPPING[tag].model_validate(params)
    return True


def check_inputs(inputs, tag):
    if tag in INPUTS_MAPPING:
        INPUTS_MAPPING[tag].model_validate(inputs)
        return True
    else:
        raise ValueError(f"{tag} is not a valid pipeline.")


AUDIO_INPUTS = {
    "automatic-speech-recognition",
    "audio-to-audio",
    "speech-segmentation",
    "audio-classification",
}
AUDIO_OUTPUTS = {
    "audio-to-audio",
    "text-to-speech",
}


IMAGE_INPUTS = {
    "image-classification",
    "image-segmentation",
    "image-to-text",
    "image-to-image",
    "object-detection",
    "zero-shot-image-classification",
}
IMAGE_OUTPUTS = {
    "image-to-image",
    "text-to-image",
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
    "text-generation",
    "text2text-generation",
    "text-classification",
    "text-to-image",
    "text-to-speech",
    "token-classification",
    "zero-shot-classification",
}


AUDIO = [
    "flac",
    "ogg",
    "mp3",
    "wav",
    "m4a",
    "aac",
    "webm",
]


IMAGE = [
    "jpeg",
    "png",
    "webp",
    "tiff",
    "bmp",
]


def parse_accept(accept: str, accepted: List[str]) -> str:
    for mimetype in accept.split(","):
        # remove quality
        mimetype = mimetype.split(";")[0]

        # remove prefix
        extension = mimetype.split("/")[-1]

        if extension in accepted:
            return extension
    return accepted[0]


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


def ffmpeg_convert(
    array: np.array, sampling_rate: int, format_for_conversion: str
) -> bytes:
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
