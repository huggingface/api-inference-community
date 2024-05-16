import datetime
import re
from abc import ABC, abstractmethod
from typing import Any

import edsnlp


class Pipeline(ABC):
    def __init__(self, model_id: str):
        self.model = edsnlp.load(model_id, auto_update=True, install_dependencies=True)

    @abstractmethod
    def __call__(self, inputs: Any) -> Any:
        raise NotImplementedError("Pipelines should implement a __call__ method")

    def parse_inputs(self, text: str):
        """
        Parse text with the following format:
        "Hello, my name is [John](PER) and I live in [New York](LOC)"
        into a Doc object with entities.
        """
        new_text = ""
        offset = 0
        ents = []
        for match in re.finditer(r"\[([^\]]*)\] *\(([^\)]*)\)", text):
            new_text = new_text + text[offset : match.start(0)]
            begin = len(new_text)
            new_text = new_text + match.group(1)
            end = len(new_text)
            offset = match.end(0)
            label = match.group(2)
            ents.append({"start": begin, "end": end, "label": label or "ent"})
        new_text = new_text + text[offset:]
        doc = self.model.make_doc(new_text)
        doc._.note_datetime = datetime.datetime.now()
        doc.ents = [
            doc.char_span(ent["start"], ent["end"], ent["label"]) for ent in ents
        ]
        return doc


class PipelineException(Exception):
    pass
