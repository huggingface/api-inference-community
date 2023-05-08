from abc import ABC, abstractmethod
from typing import Any
from optimum.intel.neural_compressor import INCModel
from transformers import Pipeline as TransformersPipeline

class Pipeline(ABC):
    _MODEL_CLASS = INCModel
    _PIPELINE_CLASS = TransformersPipeline

    @abstractmethod
    def __init__(self, model_id: str):
        self.model = self._MODEL_CLASS.from_pretrained(model_id)

    @abstractmethod
    def __call__(self, inputs: Any) -> Any:
        raise NotImplementedError("Pipelines should implement a __call__ method")


class PipelineException(Exception):
    pass
