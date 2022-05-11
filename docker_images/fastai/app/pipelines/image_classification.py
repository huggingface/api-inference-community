from typing import Any, Dict, List

import numpy as np
from app.pipelines import Pipeline
from huggingface_hub import from_pretrained_fastai
from PIL import Image


class ImageClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model = from_pretrained_fastai(model_id)

        # Obtain labels
        self.id2label = self.model.dls.vocab

        # Return at most the top 5 predicted classes
        self.top_k = 5

    def __call__(self, inputs: "Image.Image") -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`PIL.Image`):
                The raw image representation as PIL.
                No transformation made whatsoever from the input. Make all necessary transformations here.
        Return:
            A :obj:`list`:. The list contains items that are dicts should be liked {"label": "XXX", "score": 0.82}
                It is preferred if the returned list is in decreasing `score` order
        """
        # FastAI expects a np array, not a PIL Image.
        _, _, preds = self.model.predict(np.array(inputs))
        preds = preds.tolist()

        labels = [
            {"label": str(self.id2label[i]), "score": float(preds[i])}
            for i in range(len(preds))
        ]
        return sorted(labels, key=lambda tup: tup["score"], reverse=True)[: self.top_k]
