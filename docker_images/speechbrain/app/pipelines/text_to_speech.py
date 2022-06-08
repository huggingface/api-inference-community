from typing import Tuple

import numpy as np
from app.common import ModelType, get_type, get_vocoder_model_id
from app.pipelines import Pipeline
from speechbrain.pretrained import HIFIGAN, Tacotron2


class TextToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):
        model_type = get_type(model_id)
        if model_type is ModelType.TACOTRON2:
            self.model = Tacotron2.from_hparams(source=model_id)
        else:
            raise ValueError(f"{model_type.value} is invalid for text-to-speech")

        vocoder_type = get_type(model_id, "vocoder_interface")
        vocoder_model_id = get_vocoder_model_id(model_id)
        if vocoder_type is ModelType.HIFIGAN:
            self.vocoder_model = HIFIGAN.from_hparams(source=vocoder_model_id)
        else:
            raise ValueError(
                f"{vocoder_type.value} is invalid vocoder for text-to-speech"
            )

        self.sampling_rate = self.model.hparams.sample_rate

    def __call__(self, inputs: str) -> Tuple[np.array, int]:
        """
        Args:
            inputs (:obj:`str`):
                The text to generate audio from
        Return:
            A :obj:`np.array` and a :obj:`int`: The raw waveform as a numpy array, and the sampling rate as an int.
        """
        mel_output, _, _ = self.model.encode_text(inputs)
        waveforms = self.vocoder_model.decode_batch(mel_output).numpy()
        return waveforms, self.sampling_rate
