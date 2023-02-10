import json
import os
from typing import TYPE_CHECKING, Any, Dict, List

import tinyms as ts
from app.pipelines import Pipeline
from huggingface_hub import snapshot_download
from tinyms import Tensor, model, vision
from tinyms.primitives import Softmax


if TYPE_CHECKING:
    from PIL import Image


ALLOWED_MODEL = {
    "LeNet5": model.lenet5,
    "ResNet50": model.resnet50,
    "MobileNetV2": model.mobilenetv2,
}


ALLOWED_TRANSFORM = {
    "mnist": vision.mnist_transform,
    "cifar10": vision.cifar10_transform,
    "imagenet2012": vision.imagefolder_transform,
}


def load_tranform_func(config):
    dataset = config.get("dataset_transform")
    if dataset not in ALLOWED_TRANSFORM:
        raise EnvironmentError(
            f"Currently doesn't supports dataset {dataset} transform!"
        )
    return ALLOWED_TRANSFORM.get(dataset)


def load_config(config_json_file):
    with open(config_json_file, "r", encoding="utf-8") as reader:
        config = reader.read()
    return json.loads(config)


def load_model_config_from_hf(model_id):
    repo_path = snapshot_download(model_id)
    config_json_file = os.path.join(repo_path, "config.json")
    if not os.path.exists(config_json_file):
        raise EnvironmentError(
            f"The path of the config.json file {config_json_file} doesn't exist!"
        )
    config = load_config(config_json_file)
    architecture = config.get("architecture")
    if architecture not in ALLOWED_MODEL:
        raise EnvironmentError(f"Currently doesn't supports {model} model!")
    net_func = ALLOWED_MODEL.get(architecture)
    class_num = config.get("num_classes")
    net = net_func(class_num=class_num, is_training=False)
    ms_model = model.Model(net)
    model_file = os.path.join(repo_path, "mindspore_model.ckpt")
    if not os.path.exists(model_file):
        raise EnvironmentError(
            f"The path of the model file {model_file} doesn't exist!"
        )
    ms_model.load_checkpoint(model_file)
    return ms_model, config


class ImageClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model, self.config = load_model_config_from_hf(model_id)

        # Obtain labels
        self.id2label = self.config.get("id2label")

        # Get dataset transform function
        self.tranform_func = load_tranform_func(self.config)

        # Return at most the top 5 predicted classes
        self.top_k = 5

    def __call__(self, inputs: "Image.Image") -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`PIL.Image`):
                The raw image representation as PIL.
                No transformation made whatsoever
                from the input. Make all necessary transformations here.
        Return:
            A :obj:`list`:. The list contains items that are dicts should be liked {"label": "XXX", "score": 0.82}
                It is preferred if the returned list is in decreasing `score` order
        """
        # Preprocess data
        img_data = self.tranform_func(inputs)
        input_data = ts.array(img_data.tolist(), dtype=img_data.dtype.name)

        # Execute model prediction
        preds = self.model.predict(ts.expand_dims(input_data, 0))

        # Postprocess data
        softmax = Softmax()
        pred_outputs = softmax(Tensor(preds, dtype=ts.float32)).asnumpy()

        labels = [
            {"label": str(self.id2label[str(i)]), "score": float(pred_outputs[0][i])}
            for i in range(len(pred_outputs[0]))
        ]
        return sorted(labels, key=lambda tup: tup["score"], reverse=True)[: self.top_k]
