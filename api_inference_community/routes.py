import base64
import io
import logging
import os
import time
from typing import Any, Dict

from api_inference_community.validation import (
    AUDIO,
    AUDIO_INPUTS,
    IMAGE,
    IMAGE_INPUTS,
    IMAGE_OUTPUTS,
    ffmpeg_convert,
    normalize_payload,
    parse_accept,
)
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


HF_HEADER_COMPUTE_TIME = "x-compute-time"
HF_HEADER_COMPUTE_TYPE = "x-compute-type"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "cpu")

logger = logging.getLogger(__name__)


async def pipeline_route(request: Request) -> Response:
    start = time.time()
    payload = await request.body()
    task = os.environ["TASK"]
    if os.getenv("DEBUG", "0") in {"1", "true"}:
        pipe = request.app.get_pipeline()
    try:
        pipe = request.app.get_pipeline()
        try:
            sampling_rate = pipe.sampling_rate
        except Exception:
            sampling_rate = None
        inputs, params = normalize_payload(payload, task, sampling_rate=sampling_rate)
    except ValidationError as e:
        errors = []
        for error in e.errors():
            if len(error["loc"]) > 0:
                errors.append(
                    f'{error["msg"]}: received `{error["loc"][0]}` in `parameters`'
                )
            else:
                errors.append(
                    f'{error["msg"]}: received `{error["input"]}` in `parameters`'
                )
        return JSONResponse({"error": errors}, status_code=400)
    except (EnvironmentError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    accept = request.headers.get("accept", "")
    lora_adapter = request.headers.get("lora")
    if lora_adapter:
        params["lora_adapter"] = lora_adapter
    return call_pipe(pipe, inputs, params, start, accept)


def call_pipe(pipe: Any, inputs, params: Dict, start: float, accept: str) -> Response:
    root_logger = logging.getLogger()
    warnings = set()

    class RequestsHandler(logging.Handler):
        def emit(self, record):
            """Send the log records (created by loggers) to
            the appropriate destination.
            """
            warnings.add(record.getMessage())

    handler = RequestsHandler()
    handler.setLevel(logging.WARNING)
    root_logger.addHandler(handler)
    for _logger in logging.root.manager.loggerDict.values():  # type: ignore
        try:
            _logger.addHandler(handler)
        except Exception:
            pass

    status_code = 200
    if os.getenv("DEBUG", "0") in {"1", "true"}:
        outputs = pipe(inputs, **params)
    try:
        outputs = pipe(inputs, **params)
        task = os.getenv("TASK")
        metrics = get_metric(inputs, task, pipe)
    except (AssertionError, ValueError, TypeError) as e:
        outputs = {"error": str(e)}
        status_code = 400
    except Exception as e:
        outputs = {"error": "unknown error"}
        status_code = 500
        logger.error(f"There was an inference error: {e}")
        logger.exception(e)

    if warnings and isinstance(outputs, dict):
        outputs["warnings"] = list(sorted(warnings))

    compute_type = COMPUTE_TYPE
    headers = {
        HF_HEADER_COMPUTE_TIME: "{:.3f}".format(time.time() - start),
        HF_HEADER_COMPUTE_TYPE: compute_type,
        # https://stackoverflow.com/questions/43344819/reading-response-headers-with-fetch-api/44816592#44816592
        "access-control-expose-headers": f"{HF_HEADER_COMPUTE_TYPE}, {HF_HEADER_COMPUTE_TIME}",
    }

    if status_code == 200:
        headers.update(**{k: str(v) for k, v in metrics.items()})
        task = os.getenv("TASK")
        if task == "text-to-speech":
            waveform, sampling_rate = outputs
            audio_format = parse_accept(accept, AUDIO)
            data = ffmpeg_convert(waveform, sampling_rate, audio_format)
            headers["content-type"] = f"audio/{audio_format}"
            return Response(data, headers=headers, status_code=status_code)
        elif task == "audio-to-audio":
            waveforms, sampling_rate, labels = outputs
            items = []
            headers["content-type"] = "application/json"

            audio_format = parse_accept(accept, AUDIO)

            for waveform, label in zip(waveforms, labels):
                data = ffmpeg_convert(waveform, sampling_rate, audio_format)
                items.append(
                    {
                        "label": label,
                        "blob": base64.b64encode(data).decode("utf-8"),
                        "content-type": f"audio/{audio_format}",
                    }
                )
            return JSONResponse(items, headers=headers, status_code=status_code)
        elif task in IMAGE_OUTPUTS:
            image = outputs
            image_format = parse_accept(accept, IMAGE)
            buffer = io.BytesIO()
            image.save(buffer, format=image_format.upper())
            buffer.seek(0)
            img_bytes = buffer.read()
            return Response(
                img_bytes,
                headers=headers,
                status_code=200,
                media_type=f"image/{image_format}",
            )

    return JSONResponse(
        outputs,
        headers=headers,
        status_code=status_code,
    )


def get_metric(inputs, task, pipe):
    if task in AUDIO_INPUTS:
        return {"x-compute-audio-length": get_audio_length(inputs, pipe.sampling_rate)}
    elif task in IMAGE_INPUTS:
        return {"x-compute-images": 1}
    else:
        return {"x-compute-characters": get_input_characters(inputs)}


def get_audio_length(inputs, sampling_rate: int) -> float:
    if isinstance(inputs, dict):
        # Should only apply for internal AsrLive
        length_in_s = inputs["raw"].shape[0] / inputs["sampling_rate"]
    else:
        length_in_s = inputs.shape[0] / sampling_rate
    return length_in_s


def get_input_characters(inputs) -> int:
    if isinstance(inputs, str):
        return len(inputs)
    elif isinstance(inputs, (tuple, list)):
        return sum(get_input_characters(input_) for input_ in inputs)
    elif isinstance(inputs, dict):
        return sum(get_input_characters(input_) for input_ in inputs.values())
    return 0


async def status_ok(request):
    return JSONResponse({"ok": "ok"})
