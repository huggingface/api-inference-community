from typing import Dict

import numpy as np
from hezar import Model
from app.pipelines import Pipeline


class AutomaticSpeechRecognitionPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model = Model.load(model_id)
        self.sampling_rate = self.model.config.sampling_rate or 16000

    def __call__(self, inputs: np.array) -> Dict[str, str]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at self.sampling_rate, otherwise 16KHz.
        Return:
            A :obj:`dict`:. The object return should be liked {"text": "XXX"} containing
            the detected language from the input audio
        """
        outputs = self.model.predict(inputs)
        output_text = outputs[0]["text"]
        return {"text": output_text}
