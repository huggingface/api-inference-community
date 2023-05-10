from typing import Any, Dict, List

import torch
from huggingface_hub import hf_hub_download
from app.pipelines import Pipeline
from PIL import Image
from mmcv.parallel import collate, scatter

from mmcls.datasets.pipelines import Compose
from mmcls.apis import init_model

CONFIG_FILENAME = "config.py"
CHECKPOINT_FILENAME = "model.pth"


def inference_model_hf(model, img):
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        scores = model(return_loss=False, **data)
    result = [{"score": float(scores[0][i]), "label": model.CLASSES[i]} for i in range(len(model.CLASSES))]
    return sorted(result, key=lambda x: x["score"], reverse=True)


class ImageClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        config = hf_hub_download(model_id, filename=CONFIG_FILENAME)
        ckpt = hf_hub_download(model_id, filename=CHECKPOINT_FILENAME)
        self.model = init_model(config, ckpt, device="cpu")

    def __call__(self, inputs: Image.Image) -> List[Dict[str, Any]]:
        labels = inference_model_hf(self.model, inputs.filename)
        return labels
