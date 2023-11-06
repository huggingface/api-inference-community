import json
from typing import Any, Dict, List, Optional

import open_clip
import torch
import torch.nn.functional as F
from app.pipelines import Pipeline
from open_clip.pretrained import download_pretrained_from_hf
from PIL import Image


class ZeroShotImageClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            f"hf-hub:{model_id}"
        )
        config_path = download_pretrained_from_hf(
            model_id,
            filename="open_clip_config.json",
        )
        with open(config_path, "r", encoding="utf-8") as f:
            # TODO grab custom prompt templates from preprocess_cfg
            self.config = json.load(f)
        self.tokenizer = open_clip.get_tokenizer(f"hf-hub:{model_id}")
        self.model.eval()
        self.use_sigmoid = getattr(self.model, 'logit_bias', None) is not None

    def __call__(
        self,
        inputs: Image.Image,
        candidate_labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`PIL.Image`):
                The raw image representation as PIL.
                No transformation made whatsoever from the input. Make all necessary transformations here.
            candidate_labels (List[str]):
                A list of strings representing candidate class labels.
        Return:
            A :obj:`list`:. The list contains items that are dicts should be liked {"label": "XXX", "score": 0.82}
                It is preferred if the returned list is in decreasing `score` order
        """
        if candidate_labels is None:
            raise ValueError("'candidate_labels' is a required field")
        if isinstance(candidate_labels, str):
            candidate_labels = candidate_labels.split(",")

        prompt_templates = (
            "a bad photo of a {}.",
            "a photo of the large {}.",
            "art of the {}.",
            "a photo of the small {}.",
            "this is an image of {}.",
        )

        image = inputs.convert("RGB")
        image_inputs = self.preprocess(image).unsqueeze(0)

        classifier = open_clip.build_zero_shot_classifier(
            self.model,
            tokenizer=self.tokenizer,
            classnames=candidate_labels,
            templates=prompt_templates,
            num_classes_per_batch=10,
        )

        with torch.no_grad():
            image_features = self.model.encode_image(image_inputs)
            image_features = F.normalize(image_features, dim=-1)
            if self.use_sigmoid:
                logits = image_features @ classifier * self.model.logit_scale + self.model.logit_bias
                scores = torch.sigmoid(logits.squeeze(0))
            else:
                logits = image_features @ classifier * self.model.logit_scale
                scores = logits.squeeze(0).softmax(0)

        output = [
            {
                "label": l,
                "score": s.item(),
            }
            for l, s in zip(candidate_labels, scores)
        ]
        return output
