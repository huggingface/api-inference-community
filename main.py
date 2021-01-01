import time
from typing import Any, Dict

import soundfile
import uvicorn
from espnet2.bin.tts_inference import Text2Speech
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

HF_HEADER_COMPUTE_TIME = "x-compute-time"


EXAMPLE_TTS_EN_MODEL_ID = (
    "julien-c/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space_train"
)
EXAMPLE_TTS_ZH_MODEL_ID = "julien-c/kan-bayashi_csmsc_tacotron2"


MODELS: Dict[str, Any] = {}
start_time = time.time()
for model_id in (EXAMPLE_TTS_EN_MODEL_ID, EXAMPLE_TTS_ZH_MODEL_ID):
    model = Text2Speech.from_pretrained(model_id, device="cpu")
    MODELS[model_id] = model

print("models.loaded", time.time() - start_time)


def home(request: Request):
    return JSONResponse({"ok": True})


async def post_inference(request: Request):
    start = time.time()
    try:
        body = await request.json()
    except:
        return JSONResponse(status_code=400, content="Invalid JSON body")
    model_id = request.path_params["model_id"]
    print(body)
    text = body["text"]
    _model = MODELS[model_id]
    outputs = _model(text)
    speech = outputs[0]
    filename = "out-{}.wav".format(int(time.time() * 1e3))
    soundfile.write(filename, speech.numpy(), _model.fs, "PCM_16")
    return FileResponse(
        filename, headers={HF_HEADER_COMPUTE_TIME: "{:.3f}".format(time.time() - start)}
    )


routes = [
    Route("/", home),
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
    uvicorn.run(app, host="0.0.0.0", port=8000)


# curl -XPOST --data '{"text": "My name is Julien"}' http://127.0.0.1:8000/models/foo | play -
# curl -XPOST --data '{"text": "请您说得慢些好吗"}' http://127.0.0.1:8000/models/julien-c/kan-bayashi_csmsc_tacotron2 | play -
# curl -XPOST --data '{"text": "My name is Julien"}' https://api-audio.huggingface.co/models/foo | play -
