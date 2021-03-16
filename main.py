import json
import os
import tempfile
import time
from io import BytesIO
from mimetypes import guess_extension
from typing import Any, Dict, List, Optional, Tuple

import librosa
import psutil
import requests
import soundfile
import timm
import torch
import uvicorn
from asteroid import separate
from asteroid.models import BaseModel as AsteroidBaseModel
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.tts_inference import Text2Speech
from PIL import Image
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Tokenizer,
    Speech2TextForConditionalGeneration,
    Speech2TextProcessor,
)


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
    "facebook/wav2vec2-large-960h-lv60-self",
    "facebook/wav2vec2-large-xlsr-53-dutch",
    "facebook/wav2vec2-large-xlsr-53-french",
    "facebook/wav2vec2-large-xlsr-53-german",
    "facebook/wav2vec2-large-xlsr-53-italian",
    "facebook/wav2vec2-large-xlsr-53-spanish",
    "facebook/wav2vec2-large-xlsr-53-portuguese",
]

SPEECH_TO_TEXT_MODEL_IDS = [
    "facebook/s2t-small-librispeech-asr",
    "facebook/s2t-medium-librispeech-asr",
    "facebook/s2t-large-librispeech-asr",
    "facebook/s2t-small-mustc-en-de-st",
    "facebook/s2t-small-mustc-en-es-st",
    "facebook/s2t-small-mustc-en-fr-st",
    "facebook/s2t-small-mustc-en-it-st",
    "facebook/s2t-small-mustc-en-nl-st",
    "facebook/s2t-small-mustc-en-pt-st",
    "facebook/s2t-small-mustc-en-ro-st",
    "facebook/s2t-small-mustc-en-ru-st",
]

with open("data/imagenet-simple-labels.json") as f:
    IMAGENET_LABELS: List[str] = json.load(f)
## ^ from gh.com/anishathalye/imagenet-simple-labels


TTS_MODELS: Dict[str, AnyModel] = {}
ASR_MODELS: Dict[str, AnyModel] = {}
SEP_MODELS: Dict[str, AnyModel] = {}
ASR_HF_MODELS: Dict[str, Tuple[AnyModel, AnyTokenizer]] = {}
TIMM_MODELS: Dict[str, torch.nn.Module] = {}


def home(request: Request):
    return JSONResponse({"ok": True})


def health(_):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return JSONResponse(
        {**process.as_dict(attrs=["memory_percent"]), "rss": mem_info.rss}
    )


def list_models(_):
    all_models = {
        **TTS_MODELS,
        **ASR_MODELS,
        **SEP_MODELS,
        **{k: v[0] for k, v in ASR_HF_MODELS.items()},
        **TIMM_MODELS,
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

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        soundfile.write(tmp.name, speech.numpy(), model.fs, "PCM_16")

    return FileResponse(
        tmp.name,
        headers={HF_HEADER_COMPUTE_TIME: "{:.3f}".format(time.time() - start)},
        background=BackgroundTask(lambda f: os.unlink(f), tmp.name),
    )


async def post_inference_asr(
    request: Request,
    model_id: str,
):
    start = time.time()

    content_type = request.headers["content-type"].split(";")[0]

    if content_type == "application/json":
        body = await request.json()
        if "url" not in body:
            return JSONResponse(
                {"ok": False, "message": f"Invalid json, no url key"}, status_code=400
            )
        url = body["url"]
        r = requests.get(url, stream=True)
        file_ext: Optional[str] = guess_extension(
            r.headers.get("content-type", ""), strict=False
        )
        blob = r.content
    else:
        file_ext: Optional[str] = guess_extension(content_type, strict=False)
        try:
            blob = await request.body()
        except Exception as exc:
            return JSONResponse(
                {"ok": False, "message": f"Invalid body: {exc}"}, status_code=400
            )

    with tempfile.NamedTemporaryFile(suffix=file_ext) as tmp:
        print(tmp, tmp.name)
        tmp.write(blob)
        tmp.flush()

        try:
            speech, rate = soundfile.read(tmp.name, dtype="float32")
        except:
            try:
                speech, rate = librosa.load(tmp.name, sr=16_000)
            except Exception as exc:
                return JSONResponse(
                    {"ok": False, "message": f"Invalid audio: {exc}"}, status_code=400
                )

    if len(speech.shape) > 1:
        # ogg can take dual channel input -> take only first input channel in this case
        speech = speech[:, 0]
    if rate != 16_000:
        speech = librosa.resample(speech, rate, 16_000)

    ##
    ## model-specific forward pass

    if model_id in ASR_HF_MODELS:
        if model_id in SPEECH_TO_TEXT_MODEL_IDS:
            model, processor = ASR_HF_MODELS.get(model_id)
            
            inputs = processor(speech, return_tensors="pt")
            generated_ids = model.generate(input_ids=inputs["features"], attention_mask=inputs["attention_mask"])
            
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            model, tokenizer = ASR_HF_MODELS.get(model_id)

            input_values = tokenizer(speech, return_tensors="pt").input_values
            logits = model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)

            text = tokenizer.decode(predicted_ids[0])
    else:
        model = ASR_MODELS.get(model_id)
        outputs = model(speech)
        text, *_ = outputs[0]

    print(text)

    return JSONResponse(
        {"text": text},
        headers={HF_HEADER_COMPUTE_TIME: "{:.3f}".format(time.time() - start)},
    )


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

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        soundfile.write(tmp.name, est, int(model.sample_rate), "PCM_16")

    return FileResponse(
        tmp.name,
        headers={HF_HEADER_COMPUTE_TIME: "{:.3f}".format(time.time() - start)},
        background=BackgroundTask(lambda f: os.unlink(f), tmp.name),
    )


async def post_inference_timm(request: Request, model: torch.nn.Module):
    start = time.time()

    content_type = request.headers["content-type"]

    if content_type == "application/json":
        body = await request.json()
        if "url" not in body:
            return JSONResponse(
                {"ok": False, "message": f"Invalid json, no url key"}, status_code=400
            )
        url = body["url"]
        img = Image.open(requests.get(url, stream=True).raw)
    else:
        try:
            body = await request.body()
            img = Image.open(BytesIO(body))
        except Exception as exc:
            print(exc)
            return JSONResponse(
                {"ok": False, "message": f"Unable to open image from request"},
                status_code=400,
            )

    img = img.convert("RGB")

    # Data handling config
    config = model.default_cfg

    if isinstance(config["input_size"], tuple):
        img_size = config["input_size"][-2:]
    else:
        img_size = config["input_size"]

    transform = timm.data.transforms_factory.transforms_imagenet_eval(
        img_size=img_size,
        interpolation=config["interpolation"],
        mean=config["mean"],
        std=config["std"],
    )

    input_tensor = transform(img)
    input_tensor = input_tensor.unsqueeze(0)
    # ^ batch size = 1
    with torch.no_grad():
        output = model(input_tensor)

    probs = output.squeeze(0).softmax(dim=0)

    values, indices = torch.topk(probs, k=5)

    labels = [IMAGENET_LABELS[i] for i in indices]

    return JSONResponse(
        [{"label": label, "score": float(values[i])} for i, label in enumerate(labels)],
        headers={HF_HEADER_COMPUTE_TIME: "{:.3f}".format(time.time() - start)},
    )


async def post_inference(request: Request) -> JSONResponse:
    model_id = request.path_params["model_id"]

    if model_id in TTS_MODELS:
        model = TTS_MODELS.get(model_id)
        return await post_inference_tts(request, model)

    if model_id in ASR_MODELS or model_id in ASR_HF_MODELS:
        return await post_inference_asr(request, model_id)

    if model_id in SEP_MODELS:
        model = SEP_MODELS.get(model_id)
        return await post_inference_sep(request, model)

    if model_id in TIMM_MODELS:
        model = TIMM_MODELS.get(model_id)
        return await post_inference_timm(request, model)

    return JSONResponse(status_code=404, content="Unknown or unsupported model")


routes = [
    Route("/", home),
    Route("/health", health),
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
    ## Load all models

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
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_id)
        ASR_HF_MODELS[model_id] = (model, tokenizer)
    for model_id in SPEECH_TO_TEXT_MODEL_IDS:
        model = Speech2TextForConditionalGeneration.from_pretrained(model_id)
        processor = Speech2TextProcessor.from_pretrained(model_id)
        ASR_HF_MODELS[model_id] = (model, processor)

    TIMM_MODELS["julien-c/timm-dpn92"] = timm.create_model(
        "dpn92", pretrained=True
    ).eval()
    TIMM_MODELS["sgugger/resnet50d"] = timm.create_model(
        "resnet50d", pretrained=True
    ).eval()
    # ^ Those are not in eval mode by default

    print("models.loaded", time.time() - start_time)

    ## Start server
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=0)


# Sample wav file
# wget https://assets.amazon.science/c2/65/08e161cb4e96a7e007d6c3a4fef5/sample02-orig.wav


# ================
# TTS example:
# curl -XPOST --data '{"text": "My name is Julien"}' http://127.0.0.1:8000/models/julien-c/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space_train | play -
# curl -XPOST --data '{"text": "请您说得慢些好吗"}' http://127.0.0.1:8000/models/julien-c/kan-bayashi_csmsc_tacotron2 | play -
# or in production:
# curl -XPOST --data '{"text": "My name is Julien"}' http://api-audio.huggingface.co/models/julien-c/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space_train | play -

# ================
# ASR example:
# curl -i -H "Content-Type: audio/x-wav" -XPOST --data-binary '@samples/sample02-orig.wav' http://127.0.0.1:8000/models/julien-c/mini_an4_asr_train_raw_bpe_valid

# ================
# ASR wav2vec example:
# curl -i -H "Content-Type: audio/wav"  -XPOST --data-binary '@samples/sample02-orig.wav' http://127.0.0.1:8000/models/facebook/wav2vec2-base-960h
# curl -i -H "Content-Type: audio/flac" -XPOST --data-binary '@samples/sample1.flac'      http://127.0.0.1:8000/models/facebook/wav2vec2-base-960h
# curl -i -H "Content-Type: audio/webm" -XPOST --data-binary '@samples/chrome.webm'       http://127.0.0.1:8000/models/facebook/wav2vec2-base-960h
# curl -i -H "Content-Type: audio/ogg"  -XPOST --data-binary '@samples/firefox.oga'       http://127.0.0.1:8000/models/facebook/wav2vec2-base-960h

# or in production:
# curl -i -H "Content-Type: audio/wav"  -XPOST --data-binary '@samples/sample02-orig.wav' http://api-audio.huggingface.co/models/facebook/wav2vec2-base-960h

# ================
# SEP example:
# curl -XPOST --data-binary '@samples/sample02-orig.wav' http://127.0.0.1:8000/models/mhu-coder/ConvTasNet_Libri1Mix_enhsingle | play -

# ================
# TIMM examples:
# curl -i -H "Content-Type: image/jpeg"        -XPOST --data-binary '@samples/plane.jpg'       http://127.0.0.1:8000/models/julien-c/timm-dpn92
# curl -i -H "Content-Type: image/jpeg"        -XPOST --data-binary '@samples/plane.jpg'       http://127.0.0.1:8000/models/sgugger/resnet50d
# curl -i -H "Content-Type: application/json"  -XPOST --data        '{"url": "https://i.picsum.photos/id/543/536/354.jpg?hmac=O-U6guSk3J8UDMCjnqQHaL8EAOR9yHXZtgA90Bf5UTc"}'      http://127.0.0.1:8000/models/julien-c/timm-dpn92
# curl -i -H "Content-Type: application/json"  -XPOST --data        '{"url": "https://huggingface.co/front/assets/transformers-demo.png"}'      http://127.0.0.1:8000/models/sgugger/resnet50d

# or in production:
# curl -i -H "Content-Type: application/json"  -XPOST --data        '{"url": "https://huggingface.co/front/assets/transformers-demo.png"}'      http://api-audio.huggingface.co/models/sgugger/resnet50d
