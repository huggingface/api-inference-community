from typing import Dict, List

from app.common import ModelType, get_type
from app.pipelines import Pipeline
from speechbrain.pretrained import GraphemeToPhoneme


POSTPROCESSING = {ModelType.GRAPHEMETOPHONEME: lambda output: "-".join(output)}


class TextToTextPipeline(Pipeline):
    def __init__(self, model_id: str):
        model_type = get_type(model_id)
        if model_type == ModelType.GRAPHEMETOPHONEME:
            self.model = GraphemeToPhoneme.from_hparams(source=model_id)
        else:
            raise ValueError(f"{model_type.value} is invalid for text-to-text")
        self.post_process = POSTPROCESSING.get(model_type, lambda output: output)

    def __call__(self, inputs: str) -> List[Dict[str, str]]:
        """
        Args:
            inputs (:obj:`str`):
                The input text
        Return:
            A :obj:`list`:. The list contains a single item that is a dict {"text": the model output}
        """
        output = self.model(inputs)
        output = self.post_process(output)
        return [{"generated_text": output}]
