import tempfile
import time
from mimetypes import guess_extension
from typing import Any, Dict, Optional, Tuple

import librosa
import soundfile
import torch
import uvicorn
from asteroid import separate
from asteroid.models import BaseModel as AsteroidBaseModel
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.tts_inference import Text2Speech
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route
from transformers import Wav2Vec2ForMaskedLM, Wav2Vec2Tokenizer

HF_HEADER_COMPUTE_TIME = "x-compute-time"

# Type alias for all models
AnyModel = Any
AnyTokenizer = Any

EXAMPLE_TTS_EN_MODEL_ID = (
    "julien-c/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space_train"
)
EXAMPLE_TTS_ZH_MODEL_ID = "julien-c/kan-bayashi_csmsc_tacotron2"

EXAMPLE_ASR_EN_MODEL_ID = "julien-c/mini_an4_asr_train_raw_bpe_valid"

EXAMPLE_SEP_ENH_MODEL_ID = "mhu-coder/ConvTasNet_Libri1Mix_enhsingle"
EXAMPLE_SEP_SEP_MODEL_ID = "julien-c/DPRNNTasNet-ks16_WHAM_sepclean"

WAV2VEV2_MODEL_IDS = [
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-large-960h",
    "facebook/wav2vec2-large-960h-lv60",
    "facebook/wav2vec2-large-960h-lv60-self",
]


TTS_MODELS: Dict[str, AnyModel] = {}
ASR_MODELS: Dict[str, AnyModel] = {}
SEP_MODELS: Dict[str, AnyModel] = {}
ASR_HF_MODELS: Dict[str, Tuple[AnyModel, AnyTokenizer]] = {}

start_time = time.time()
for model_id in (EXAMPLE_TTS_EN_MODEL_ID, EXAMPLE_TTS_ZH_MODEL_ID):
    model = Text2Speech.from_pretrained(model_id, device="cpu")
    TTS_MODELS[model_id] = model
for model_id in (EXAMPLE_ASR_EN_MODEL_ID,):
    model = Speech2Text.from_pretrained(model_id, device="cpu")
    ASR_MODELS[model_id] = model
for model_id in (EXAMPLE_SEP_ENH_MODEL_ID, EXAMPLE_SEP_SEP_MODEL_ID):
    model = AsteroidBaseModel.from_pretrained(model_id)
    SEP_MODELS[model_id] = model
for model_id in WAV2VEV2_MODEL_IDS:
    model = Wav2Vec2ForMaskedLM.from_pretrained(model_id)
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_id)
    ASR_HF_MODELS[model_id] = (model, tokenizer)

print("models.loaded", time.time() - start_time)


def home(request: Request):
    return JSONResponse({"ok": True})


def list_models(_):
    all_models = {
        **TTS_MODELS,
        **ASR_MODELS,
        **SEP_MODELS,
        **{k: v[0] for k, v in ASR_HF_MODELS.items()},
    }
    return JSONResponse({k: v.__class__.__name__ for k, v in all_models.items()})


async def post_inference_tts(request: Request, model: AnyModel):
    start = time.time()

    try:
        body = await request.json()
    except:
        return JSONResponse(status_code=400, content="Invalid JSON body")
    print(body)
    text = body["text"]

    outputs = model(text)
    speech = outputs[0]
    filename = "out-{}.wav".format(int(time.time() * 1e3))
    soundfile.write(filename, speech.numpy(), model.fs, "PCM_16")
    return FileResponse(
        filename, headers={HF_HEADER_COMPUTE_TIME: "{:.3f}".format(time.time() - start)}
    )


async def post_inference_asr(request: Request, model: AnyModel):
    start = time.time()

    print(request.headers)
    content_type = request.headers["content-type"]
    print(content_type)
    file_ext: Optional[str] = guess_extension(content_type, strict=False)
    print(file_ext)

    try:
        body = await request.body()
        with tempfile.NamedTemporaryFile(suffix=file_ext) as tmp:
            print(tmp, tmp.name)
            tmp.write(body)
            tmp.flush()
            speech, rate = soundfile.read(tmp.name)
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "message": f"Invalid body: {exc}"}, status_code=400
        )

    outputs = model(speech)
    text, *_ = outputs[0]
    print(text)

    return JSONResponse({"text": text})


async def post_inference_asr_hf(
    request: Request, model: AnyModel, tokenizer: AnyTokenizer
):
    start = time.time()

    print(request.headers)
    content_type = request.headers["content-type"]
    print(content_type)
    file_ext: Optional[str] = guess_extension(content_type, strict=False)
    print(file_ext)

    try:
        body = await request.body()
        with tempfile.NamedTemporaryFile(suffix=file_ext) as tmp:
            print(tmp, tmp.name)
            tmp.write(body)
            tmp.flush()
            speech, rate = soundfile.read(tmp.name)
            if rate < 16000:
                return JSONResponse(
                    {
                        "ok": False,
                        "message": f"Invalid sampling rate of file. Make sure the uploaded audio file was sampled at 16000 Hz or higher, not {rate} Hz",
                    },
                    status_code=400,
                )
            elif rate > 16000:
                speech = librosa.resample(speech, rate, 16000)
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "message": f"Invalid body: {exc}"}, status_code=400
        )

    input_values = tokenizer(speech, return_tensors="pt").input_values
    logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    text = tokenizer.decode(predicted_ids[0])
    print(text)

    return JSONResponse({"text": text})


async def post_inference_sep(request: Request, model: AnyModel):
    start = time.time()

    try:
        body = await request.body()
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(body)
            tmp.flush()
            wav, fs = separate._load_audio(tmp.name)
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "message": f"Invalid body: {exc}"}, status_code=400
        )

    # Wav shape: [time, n_chan]
    # We only support n_chan = 1 for now.
    wav = separate._resample(wav[:, 0], orig_sr=fs, target_sr=int(model.sample_rate))
    # Pass wav as [batch, n_chan, time]; here: [1, 1, time]
    (est_srcs,) = separate.numpy_separate(model, wav.reshape((1, 1, -1)))
    # FIXME: how to deal with multiple sources?
    est = est_srcs[0]

    filename = "out-{}.wav".format(int(time.time() * 1e3))
    soundfile.write(filename, est, int(model.sample_rate), "PCM_16")
    return FileResponse(
        filename, headers={HF_HEADER_COMPUTE_TIME: "{:.3f}".format(time.time() - start)}
    )


async def post_inference(request: Request) -> JSONResponse:
    model_id = request.path_params["model_id"]

    if model_id in TTS_MODELS:
        model = TTS_MODELS.get(model_id)
        return await post_inference_tts(request, model)

    if model_id in ASR_MODELS:
        model = ASR_MODELS.get(model_id)
        return await post_inference_asr(request, model)

    if model_id in ASR_HF_MODELS:
        model, tokenizer = ASR_HF_MODELS.get(model_id)
        return await post_inference_asr_hf(request, model, tokenizer)

    if model_id in SEP_MODELS:
        model = SEP_MODELS.get(model_id)
        return await post_inference_sep(request, model)

    return JSONResponse(status_code=404, content="Unknown or unsupported model")


routes = [
    Route("/", home),
    Route("/models", list_models),
    Route("/models/{model_id:path}", post_inference, methods=["POST"]),
]

middlewares = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
]

app = Starlette(debug=True, routes=routes, middleware=middlewares)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=0)


# Sample wav file
# wget https://assets.amazon.science/c2/65/08e161cb4e96a7e007d6c3a4fef5/sample02-orig.wav


# TTS example:
# curl -XPOST --data '{"text": "My name is Julien"}' http://127.0.0.1:8000/models/julien-c/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space_train | play -
# curl -XPOST --data '{"text": "请您说得慢些好吗"}' http://127.0.0.1:8000/models/julien-c/kan-bayashi_csmsc_tacotron2 | play -
# or in production:
# curl -XPOST --data '{"text": "My name is Julien"}' http://api-audio.huggingface.co/models/julien-c/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space_train | play -

# ASR example:
# curl -i -H "Content-Type: audio/wav" -XPOST --data-binary '@sample02-orig.wav' http://127.0.0.1:8000/models/julien-c/mini_an4_asr_train_raw_bpe_valid

# ASR wav2vec example:
# curl -i -H "Content-Type: audio/wav" -XPOST --data-binary '@sample02-orig.wav' http://127.0.0.1:8000/models/facebook/wav2vec2-base-960h

# SEP example:
# curl -XPOST --data-binary '@sample02-orig.wav' http://127.0.0.1:8000/models/mhu-coder/ConvTasNet_Libri1Mix_enhsingle | play -
