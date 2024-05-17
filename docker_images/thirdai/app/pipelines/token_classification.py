from typing import Any, Dict, List

from app.pipelines import Pipeline
from thirdai import bolt


class TokenClassificationPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        self.model = bolt.UniversalDeepTransformer.NER.load(model_id)

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
        split_inputs = inputs.split(" ")

        outputs = self.model.predict(split_inputs)

        entities = []
        offset = 0
        for entity_results, word in zip(outputs, split_inputs):
            best_prediction = entity_results[0]

            current_entity = {
                "entity_group": best_prediction[0],
                "word": word,
                "start": offset,
                "end": offset + len(word),
                "score": best_prediction[1],
            }

            entities.append(current_entity)

            offset += len(word) + 1
    
        return entities