from typing import Dict, List

import numpy as np
import torch
from app.common import ModelType, get_type
from app.pipelines import Pipeline
from speechbrain.pretrained import Speech_Emotion_Diarization


class VoiceActivityDetectionPipeline(Pipeline):
    def __init__(self, model_id: str):
        model_type = get_type(model_id)
        if model_type == ModelType.SPEECHEMOTIONDIARIZATION:
            self.model = Speech_Emotion_Diarization.from_hparams(source=model_id)
        else:
            raise ValueError(f"{model_type.value} is invalid for voice-activity-detection")

        # Please define a `self.sampling_rate` for this pipeline
        # to automatically read the input correctly
        self.sampling_rate = 16000

    def __call__(self, inputs: np.array) -> Dict[str, List[Dict[str, float]]]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at 16KHz.
        Return:
            A :obj:`dict`:. The object returned should be a dictionary like {"audio_id": [{"start": 0.0, "end": 2.0, "emotion": "n"}]} containing :
                - "start": A float representing the starting timestamp of a speech event.
                - "end": A float representing the ending timestamp of a speech event.
                - "emotion": A str representing the content of the speech event.
        """
        batch = torch.from_numpy(inputs).unsqueeze(0)
        rel_length = torch.tensor([1.0])
        results = model.diarize_batch(batch, rel_length, ["hf_api_test_audio"])
        return results