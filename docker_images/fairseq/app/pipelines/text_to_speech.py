import logging
import os
from typing import Tuple

import numpy as np
import torch

from app.pipelines import Pipeline

logger = logging.getLogger(__name__)


class TextToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):

        logger.info("Start to load model")

        self.model = torch.hub.load("pytorch/fairseq:main", model_id)
        self.sampling_rate = 16000

        logger.info("Load model successfully")

    def __call__(self, inputs: str) -> Tuple[np.array, int]:
        """
        Args:
            inputs (:obj:`str`):
                The text to generate audio from
        Return:
            A :obj:`np.array` and a :obj:`int`: The raw waveform as a numpy
            array, and the sampling rate as an int.
        """

        logger.info("Start to predict")

        inputs = inputs.strip("\x00")
        if len(inputs) == 0:
            return np.zeros((0,)), self.sampling_rate
        waveform, sample_rate = self.model.predict(inputs)

        logger.info("Finish predicting")

        return waveform.numpy(), sample_rate
