import numpy as np
from app.common import ModelType, get_type
from app.pipelines import Pipeline
from speechbrain.pretrained import GraphemeToPhoneme

class GraphemeToPhonemePipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model = GraphemeToPhoneme.from_hparams(source=model_id)

    def __call__(self, inputs: str) -> str:
        """
        Args:
            inputs (:obj:`str`):
                The text to generate audio from
        Return:
            A :obj:`np.array` and a :obj:`int`: The raw waveform as a numpy array, and the sampling rate as an int.
        """
        phn = self.model(inputs)
        phn_txt = " ".join(phn)
        return {"phn": phn_txt}