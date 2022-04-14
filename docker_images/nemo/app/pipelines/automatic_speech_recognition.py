import os
import tempfile
import uuid
from typing import Dict

import librosa
import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile
from app.pipelines import Pipeline
from huggingface_hub import hf_hub_download
from huggingface_hub.hf_api import HfFolder


class AutomaticSpeechRecognitionPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here

        # Precheck for API key
        is_token_available = HfFolder.get_token() is not None

        # Prepare file name from model_id
        filename = model_id.split("/")[-1] + ".nemo"
        path = hf_hub_download(
            repo_id=model_id, filename=filename, use_auth_token=is_token_available
        )

        # Load model
        self.model = nemo_asr.models.ASRModel.restore_from(path)
        self.model.freeze()

        # Pre-Initialize RNNT decoding strategy
        if hasattr(self.model, "change_decoding_strategy"):
            self.model.change_decoding_strategy(None)

        # IMPLEMENT_THIS : Please define a `self.sampling_rate` for this pipeline
        # to automatically read the input correctly
        self.sampling_rate = self.model.cfg.sample_rate

    def __call__(self, inputs: np.array) -> Dict[str, str]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at self.sampling_rate, otherwise 16KHz.
        Return:
            A :obj:`dict`:. The object return should be liked {"text": "XXX"} containing
            the detected langage from the input audio
        """
        inputs = self.process_audio_file(inputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, f"audio_{uuid.uuid4()}.wav")
            soundfile.write(audio_path, inputs, self.sampling_rate)

            transcriptions = self.model.transcribe([audio_path])

            # if transcriptions form a tuple (from RNNT), extract just "best" hypothesis
            if type(transcriptions) == tuple and len(transcriptions) == 2:
                transcriptions = transcriptions[0]

        audio_transcription = transcriptions[0]

        return {"text": audio_transcription}

    def process_audio_file(self, data):
        # monochannel
        data = librosa.to_mono(data)
        return data
