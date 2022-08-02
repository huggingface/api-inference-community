import os
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS, get_pipeline


# Must contain at least one example of each implemented pipeline
TESTABLE_MODELS = {
    "tabular-classification": {
        "iris-sklearn-latest-without-config": {
            "input": "iris-latest-input.json",
            "output": "iris-latest-output.json",
            "has_config": False,
            "old_sklearn": False,
        },
        "iris-sklearn-latest-with-config": {
            "input": "iris-latest-input.json",
            "output": "iris-latest-output.json",
            "has_config": True,
            "old_sklearn": False,
        },
        "iris-sklearn-1.0-without-config": {
            "input": "iris-1.0-input.json",
            "output": "iris-1.0-output.json",
            "has_config": False,
            "old_sklearn": True,
        },
        "iris-sklearn-1.0-with-config": {
            "input": "iris-1.0-input.json",
            "output": "iris-1.0-output.json",
            "has_config": True,
            "old_sklearn": True,
        },
    }
}

ALL_TASKS = {
    "automatic-speech-recognition",
    "audio-source-separation",
    "feature-extraction",
    "image-classification",
    "question-answering",
    "sentence-similarity",
    "tabular-classification",
    "text-generation",
    "text-to-speech",
    "token-classification",
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
