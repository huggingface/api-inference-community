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
            A :obj:`list`:. The object returned should be a list of one list like [[{"label": "positive", "score": 0.5}]] containing:
                - "label": A string representing what the label/class is. There can be multiple labels.
                - "score": A score between 0 and 1 describing how confident the model is for this label/class.
        """
        topics, probabilities = self.model.transform(inputs)
        results = []
        for topic, prob in zip(topics, probabilities):
            if self.model.custom_labels_ is not None:
                topic_label = self.model.custom_labels_[topic + self.model_outliers]
           else:
                topic_label = self.model.topic_labels_[topic]
            results.append({"label": topic_label, "score": float(prob)})
        return [results]
