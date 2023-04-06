from typing import Any, Dict, List

from app.pipelines import Pipeline
from span_marker import SpanMarkerModel


class TokenClassificationPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ) -> None:
        self.model = SpanMarkerModel.from_pretrained(model_id)

    def __call__(self, inputs: str) -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`list`:. The object returned should be like [{"entity_group": "XXX", "word": "some word", "start": 3, "end": 6, "score": 0.82}] containing :
                - "entity_group": A string representing what the entity is.
                - "word": A rubstring of the original string that was detected as an entity.
                - "start": the offset within `input` leading to `answer`. context[start:stop] == word
                - "end": the ending offset within `input` leading to `answer`. context[start:stop] === word
                - "score": A score between 0 and 1 describing how confident the model is for this entity.
        """
        return [
            {
                "entity_group": entity["label"],
                "word": entity["span"],
                "start": entity["char_start_index"],
                "end": entity["char_end_index"],
                "score": entity["score"],
            }
            for entity in self.model.predict(inputs)
        ]
