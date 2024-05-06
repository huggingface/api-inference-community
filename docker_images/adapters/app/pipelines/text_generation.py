from typing import Dict, List

from app.pipelines import Pipeline
from transformers import TextGenerationPipeline as TransformersTextGenerationPipeline


class TextGenerationPipeline(Pipeline):
    def __init__(self, adapter_id: str):
        self.pipeline = self._load_pipeline_instance(
            TransformersTextGenerationPipeline, adapter_id
        )

    def __call__(self, inputs: str) -> List[Dict[str, str]]:
        """
        Args:
            inputs (:obj:`str`):
                The input text
        Return:
            A :obj:`list`:. The list contains a single item that is a dict {"text": the model output}
        """
        return self.pipeline(inputs)
