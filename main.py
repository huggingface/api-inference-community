import tempfile
import time
from typing import Any, Dict

import soundfile
import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from espnet2.bin.tts_inference import Text2Speech
from asteroid.models import BaseModel as AsteroidBaseModel
from asteroid import separate


HF_HEADER_COMPUTE_TIME = "x-compute-time"


EXAMPLE_TTS_EN_MODEL_ID = (
    "julien-c/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space_train"
)
EXAMPLE_TTS_ZH_MODEL_ID = "julien-c/kan-bayashi_csmsc_tacotron2"
EXAMPLE_SEP_ENH_MODEL_ID = "mhu-coder/ConvTasNet_Libri1Mix_enhsingle"
EXAMPLE_SEP_SEP_MODEL_ID = "julien-c/DPRNNTasNet-ks16_WHAM_sepclean"


TTS_MODELS: Dict[str, Any] = {}
SEP_MODELS: Dict[str, Any] = {}
start_time = time.time()
for model_id in (EXAMPLE_TTS_EN_MODEL_ID, EXAMPLE_TTS_ZH_MODEL_ID):
    model = Text2Speech.from_pretrained(model_id, device="cpu")
    TTS_MODELS[model_id] = model
for model_id in (EXAMPLE_SEP_ENH_MODEL_ID, EXAMPLE_SEP_SEP_MODEL_ID):
    model = AsteroidBaseModel.from_pretrained(model_id)
    SEP_MODELS[model_id] = model

print("models.loaded", time.time() - start_time)


def home(request: Request):
    return JSONResponse({"ok": True})


async def post_inference_tts(request: Request):
    start = time.time()
    try:
        body = await request.json()
    except:
        return JSONResponse(status_code=400, content="Invalid JSON body")
    model_id = request.path_params["model_id"]
    print(body)
    text = body["text"]
    _model = TTS_MODELS[model_id]
    outputs = _model(text)
    speech = outputs[0]
    filename = "out-{}.wav".format(int(time.time() * 1e3))
    soundfile.write(filename, speech.numpy(), _model.fs, "PCM_16")
    return FileResponse(
        filename, headers={HF_HEADER_COMPUTE_TIME: "{:.3f}".format(time.time() - start)}
    )


async def post_inference_sep(request: Request):
    start = time.time()

    model_id = request.path_params["model_id"]
    model = SEP_MODELS[model_id]

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


routes = [
    Route("/", home),
    Route("/tts/{model_id:path}", post_inference_tts, methods=["POST"]),
    Route("/sep/{model_id:path}", post_inference_sep, methods=["POST"]),
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
    uvicorn.run(app, host="0.0.0.0", port=8000)


# TTS example:
# curl -XPOST --data '{"text": "My name is Julien"}' http://127.0.0.1:8000/tts/foo | play -
# curl -XPOST --data '{"text": "请您说得慢些好吗"}' http://127.0.0.1:8000/tts/julien-c/kan-bayashi_csmsc_tacotron2 | play -
# curl -XPOST --data '{"text": "My name is Julien"}' https://api-audio.huggingface.co/tts/foo | play -

# Seperation example:
# wget https://assets.amazon.science/c2/65/08e161cb4e96a7e007d6c3a4fef5/sample02-orig.wav
# curl -XPOST --data-binary '@sample02-orig.wav' http://127.0.0.1:8000/sep/mhu-coder/ConvTasNet_Libri1Mix_enhsingle | play -
