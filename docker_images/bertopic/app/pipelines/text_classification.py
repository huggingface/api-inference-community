from typing import Dict, List

from app.pipelines import Pipeline
from bertopic import BERTopic


class TextClassificationPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        self.model = BERTopic.load(model_id)

    def __call__(self, inputs: str) -> List[List[Dict[str, float]]]:
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`list`:. The object returned should be a list of one list like [[{"label": 0.9939950108528137}]] containing:
                - "label": A string representing what the label/class is. There can be multiple labels.
                - "score": A score between 0 and 1 describing how confident the model is for this label/class.
        """
        topics, probs = self.model.transform(inputs)
        topic_label = self.model.generate_topic_labels()[topics[0]]
        return [[{"label": topic_label, "score": float(probs[0])}]]
