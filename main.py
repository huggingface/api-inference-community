import time

import soundfile
import uvicorn
from espnet2.bin.tts_inference import Text2Speech
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse
from starlette.routing import Route

EXAMPLE_TTS_EN_MODEL_ID = (
    "julien-c/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space_train"
)


model = Text2Speech.from_pretrained(
    EXAMPLE_TTS_EN_MODEL_ID,
    device="cpu"
)

print("model.loaded")


async def post_inference(request: Request):
    body = await request.json()
    print(body)
    outputs = model(body["text"])
    speech = outputs[0]
    filename = "out-{}.wav".format(int(time.time() * 1e3))
    soundfile.write(filename, speech.numpy(), model.fs, "PCM_16")
    return FileResponse(filename)


routes = [
    Route('/models/{model_id}', post_inference, methods=["POST"]),
]

app = Starlette(debug=True, routes=routes)

if __name__ == "__main__":
    uvicorn.run(app)


# curl -XPOST --data '{"text": "My name is Julien"}' http://127.0.0.1:8000/models/foo | play -
# curl -XPOST --data '{"text": "My name is Julien"}' https://api-audio.huggingface.co/models/foo | play -