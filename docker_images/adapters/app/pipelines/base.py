from abc import ABC, abstractmethod
from typing import Any

from adapters import AutoAdapterModel, get_adapter_info
from transformers import AutoTokenizer
from transformers.pipelines.base import logger


class Pipeline(ABC):
    @abstractmethod
    def __init__(self, model_id: str):
        raise NotImplementedError("Pipelines should implement an __init__ method")

    @abstractmethod
    def __call__(self, inputs: Any) -> Any:
        raise NotImplementedError("Pipelines should implement a __call__ method")

    @staticmethod
    def _load_pipeline_instance(pipeline_class, adapter_id):
        adapter_info = get_adapter_info(adapter_id, source="hf")
        if adapter_info is None:
            raise ValueError(f"Adapter with id '{adapter_id}' not available.")

        tokenizer = AutoTokenizer.from_pretrained(adapter_info.model_name)
        model = AutoAdapterModel.from_pretrained(adapter_info.model_name)
        model.load_adapter(adapter_id, source="hf", set_active=True)

        # Transformers incorrectly logs an error because class name is not known. Filter this out.
        logger.addFilter(
            lambda record: not record.getMessage().startswith(f"The model '{model.__class__.__name__}' is not supported")
        )

        return pipeline_class(model=model, tokenizer=tokenizer)


class PipelineException(Exception):
    pass
