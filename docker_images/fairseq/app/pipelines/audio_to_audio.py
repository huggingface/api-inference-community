import logging
import os
from typing import List, Tuple

import numpy as np
import torch

from app.pipelines import Pipeline

logger = logging.getLogger(__name__)

class SpeechToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):

        logger.info("Start to load model")

        self.model = torch.hub.load("pytorch/fairseq:main", model_id, generation_args={'beam': 1, 'max_len_a': 0.003125})
        self.sampling_rate = 16000

        logger.info("Load model successfully")

    def __call__(self, inputs: np.array) -> Tuple[np.array, int, List[str]]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default sampled at `self.sampling_rate`.
                The shape of this array is `T`, where `T` is the time axis
        Return:
            A :obj:`tuple` containing:
              - :obj:`np.array`:
                 The return shape of the array must be `C'`x`T'`
              - a :obj:`int`: the sampling rate as an int in Hz.
              - a :obj:`List[str]`: the annotation for each out channel.
                    This can be the name of the instruments for audio source separation
                    or some annotation for speech enhancement. The length must be `C'`.
        """

        logger.info("Start to predict")

        _inputs = torch.from_numpy(inputs).unsqueeze(0)
        units, audio = self.model.predict(_inputs, synthesize_speech=True)
        waveform, sample_rate = audio

        logger.info("Finish predicting")

        return waveform.unsqueeze(0).numpy(), sample_rate, [units]
