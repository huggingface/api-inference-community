from typing import Dict, List

from app.pipelines import Pipeline
from thirdai import bolt
import numpy as np


class TextClassificationPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        self.model = bolt.UniversalDeepTransformer.load(model_id)

    def __call__(self, inputs: str) -> List[Dict[str, float]]:
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`list`:. The object returned should be a list of one list like [[{"label": 0.9939950108528137}]] containing:
                - "label": A string representing what the label/class is. There can be multiple labels.
                - "score": A score between 0 and 1 describing how confident the model is for this label/class.
        """
        outputs = self.model.predict({self.model.text_dataset_config().text_column: inputs})

        if len(outputs) == 0:
            return []
        
        if isinstance(outputs[0], tuple):
            return [[{str(outputs[0][0]): outputs[0][1]}]]
        else:
            index = np.argmax(outputs)
            return [[{self.model.class_name(index): outputs[index]}]]
