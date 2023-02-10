from typing import Any, Dict, List

import timm
import torch
from app.pipelines import Pipeline
from PIL import Image
from timm.data import (
    CustomDatasetInfo,
    ImageNetInfo,
    create_transform,
    infer_imagenet_subset,
    resolve_model_data_config,
)


class ImageClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model = timm.create_model(f"hf_hub:{model_id}", pretrained=True)
        self.transform = create_transform(
            **resolve_model_data_config(self.model, use_test_size=True)
        )
        self.top_k = min(self.model.num_classes, 5)
        self.model.eval()

        self.dataset_info = None
        label_names = self.model.pretrained_cfg.get("label_names", None)
        label_descriptions = self.model.pretrained_cfg.get("label_descriptions", None)

        if label_names is None:
            # if no labels added to config, use imagenet labeller in timm
            imagenet_subset = infer_imagenet_subset(self.model)
            if imagenet_subset:
                self.dataset_info = ImageNetInfo(imagenet_subset)
            else:
                # fallback label names
                label_names = [f"LABEL_{i}" for i in range(self.model.num_classes)]

        if self.dataset_info is None:
            self.dataset_info = CustomDatasetInfo(
                label_names=label_names,
                label_descriptions=label_descriptions,
            )

    def __call__(self, inputs: Image.Image) -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`PIL.Image`):
                The raw image representation as PIL.
                No transformation made whatsoever from the input. Make all necessary transformations here.
        Return:
            A :obj:`list`:. The list contains items that are dicts should be liked {"label": "XXX", "score": 0.82}
                It is preferred if the returned list is in decreasing `score` order
        """
        im = inputs.convert("RGB")
        inputs = self.transform(im).unsqueeze(0)

        with torch.no_grad():
            out = self.model(inputs)

        probabilities = out.squeeze(0).softmax(dim=0)
        values, indices = torch.topk(probabilities, self.top_k)

        labels = [
            {
                "label": self.dataset_info.index_to_description(i, detailed=True),
                "score": v.item(),
            }
            for i, v in zip(indices, values)
        ]
        return labels
