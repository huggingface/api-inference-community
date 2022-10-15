import os
from typing import Dict, List
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS, get_pipeline


# Must contain at least one example of each implemented pipeline
# Tests do not check the actual values of the model output, so small dummy
# models are recommended for faster tests.
TESTABLE_MODELS: Dict[str, List[str]] = {
    "audio-classification": [
        # Language Identification
        "speechbrain/lang-id-commonlanguage_ecapa",
        # Command recognition
        "speechbrain/google_speech_command_xvector",
        # Speaker recognition
        "speechbrain/spkrec-xvect-voxceleb",
    ],
    "audio-to-audio": [
        # Speech Enhancement
        "speechbrain/mtl-mimic-voicebank",
        # Source separation
        "speechbrain/sepformer-wham",
    ],
    "automatic-speech-recognition": [
        # ASR with EncoderASR
        "speechbrain/asr-wav2vec2-commonvoice-fr",
        # ASR with EncoderDecoderASR
        "speechbrain/asr-crdnn-commonvoice-it",
    ],
    "text-to-speech": [
        "speechbrain/tts-tacotron2-ljspeech",
    ],
    "text2text-generation": [
        # SoundChoice G2P
        "speechbrain/soundchoice-g2p"
    ],
}


ALL_TASKS = {
    "audio-classification",
    "audio-to-audio",
    "automatic-speech-recognition",
    "audio-source-separation",
    "image-classification",
    "question-answering",
    "text-generation",
    "text-to-speech",
}


class PipelineTestCase(TestCase):
    @skipIf(
        os.path.dirname(os.path.dirname(__file__)).endswith("common"),
        "common is a special case",
    )
    def test_has_at_least_one_task_enabled(self):
        self.assertGreater(
            len(ALLOWED_TASKS.keys()), 0, "You need to implement at least one task"
        )

    def test_unsupported_tasks(self):
        unsupported_tasks = ALL_TASKS - ALLOWED_TASKS.keys()
        for unsupported_task in unsupported_tasks:
            with self.subTest(msg=unsupported_task, task=unsupported_task):
                with self.assertRaises(EnvironmentError):
                    get_pipeline(unsupported_task, model_id="XX")
