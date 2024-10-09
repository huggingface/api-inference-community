import os
from typing import Dict
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS, get_pipeline


# Must contain at least one example of each implemented pipeline
# Tests do not check the actual values of the model output, so small dummy
# models are recommended for faster tests.
TESTABLE_MODELS: Dict[str, str] = {
    "zero-shot-image-classification": [
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        # "laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg",
        # "timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k",
    ]
}


ALL_TASKS = {
    "audio-classification",
    "audio-to-audio",
    "automatic-speech-recognition",
    "feature-extraction",
    "image-classification",
    "question-answering",
    "speech-segmentation",
    "tabular-classification",
    "tabular-regression",
    "text-classification",
    "text-to-image",
    "text-to-speech",
    "token-classification",
    "conversational",
    "feature-extraction",
    "question-answering",
    "fill-mask",
    "table-question-answering",
    "summarization",
    "text2text-generation",
    "text-classification",
    "text-to-image",
    "text-to-speech",
    "token-classification",
    "zero-shot-classification",
    "zero-shot-image-classification",
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
                os.environ["TASK"] = unsupported_task
                os.environ["MODEL_ID"] = "XX"
                with self.assertRaises(EnvironmentError):
                    get_pipeline()
