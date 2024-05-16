from typing import Any, Dict, List

from app.pipelines import Pipeline


class TokenClassificationPipeline(Pipeline):
    def __call__(self, inputs: str) -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`list`:. The object returned should be like [{"entity_group": "XXX", "word": "some word", "start": 3, "end": 6, "score": 0.82}] containing :
                - "entity_group": A string representing what the entity is.
                - "word": A substring of the original string that was detected as an entity.
                - "start": the offset within `input` leading to `answer`. context[start:stop] == word
                - "end": the ending offset within `input` leading to `answer`. context[start:stop] === word
                - "score": A score between 0 and 1 describing how confident the model is for this entity.
                - "value": An optional value for the entity (date, amount, etc.)
        """
        doc = self.parse_inputs(inputs)
        doc = self.model(doc)

        entities = []
        for ent in doc.ents:
            current_entity = {
                "entity_group": ent.label_,
                "word": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "score": 1.0,
                "value": ent._.value,
            }
            entities.append(current_entity)

        return entities
