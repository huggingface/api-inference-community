import os
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS, get_pipeline


# Must contain at least one example of each implemented pipeline
TESTABLE_MODELS = {
    "tabular-classification": [
        "skops-tests/iris-sklearn-1.0-logistic_regression-with-config",
        "skops-tests/iris-sklearn-1.0-logistic_regression-without-config",
        "skops-tests/iris-sklearn-1.0-hist_gradient_boosting-with-config",
        "skops-tests/iris-sklearn-1.0-hist_gradient_boosting-without-config",
        "skops-tests/iris-sklearn-latest-logistic_regression-with-config",
        "skops-tests/iris-sklearn-latest-logistic_regression-without-config",
        "skops-tests/iris-sklearn-latest-hist_gradient_boosting-with-config",
        "skops-tests/iris-sklearn-latest-hist_gradient_boosting-without-config",
    ],
    "tabular-regression": [
        "skops-tests/tabularregression-sklearn-1.0-linear_regression-with-config",
        "skops-tests/tabularregression-sklearn-1.0-linear_regression-without-config",
        "skops-tests/tabularregression-sklearn-1.0-hist_gradient_boosting_regressor-with-config",
        "skops-tests/tabularregression-sklearn-1.0-hist_gradient_boosting_regressor-without-config",
        "skops-tests/tabularregression-sklearn-latest-linear_regression-with-config",
        "skops-tests/tabularregression-sklearn-latest-linear_regression-without-config",
        "skops-tests/tabularregression-sklearn-latest-hist_gradient_boosting_regressor-with-config",
        "skops-tests/tabularregression-sklearn-latest-hist_gradient_boosting_regressor-without-config",
    ],
    "text-classification": [
        "skops-tests/textclassification-sklearn-hist_gradient_boosting-latest-without-config",
        "skops-tests/textclassification-sklearn-hist_gradient_boosting-latest-with-config",
        "skops-tests/textclassification-sklearn-hist_gradient_boosting-1.0-without-config",
        "skops-tests/textclassification-sklearn-hist_gradient_boosting-1.0-with-config",
        "skops-tests/textclassification-sklearn-logistic_regression-latest-without-config",
        "skops-tests/textclassification-sklearn-logistic_regression-latest-with-config",
        "skops-tests/textclassification-sklearn-logistic_regression-1.0-without-config",
        "skops-tests/textclassification-sklearn-logistic_regression-1.0-with-config",
    ],
}

# This contains information about the test cases above, used in the tests to
# define which tests to run for which examples.
TEST_CASES = {
    "tabular-classification": {
        "skops-tests/iris-sklearn-latest-logistic_regression-without-config": {
            "input": "iris-latest-input.json",
            "output": "iris-logistic_regression-latest-output.json",
            "has_config": False,
            "old_sklearn": False,
            "accepts_nan": False,
            "loads_on_new_sklearn": True,
        },
        "skops-tests/iris-sklearn-latest-logistic_regression-with-config": {
            "input": "iris-latest-input.json",
            "output": "iris-logistic_regression-latest-output.json",
            "has_config": True,
            "old_sklearn": False,
            "accepts_nan": False,
            "loads_on_new_sklearn": True,
        },
        "skops-tests/iris-sklearn-1.0-logistic_regression-without-config": {
            "input": "iris-1.0-input.json",
            "output": "iris-logistic_regression-1.0-output.json",
            "has_config": False,
            "old_sklearn": True,
            "accepts_nan": False,
            "loads_on_new_sklearn": True,
        },
        "skops-tests/iris-sklearn-1.0-logistic_regression-with-config": {
            "input": "iris-1.0-input.json",
            "output": "iris-logistic_regression-1.0-output.json",
            "has_config": True,
            "old_sklearn": True,
            "accepts_nan": False,
            "loads_on_new_sklearn": True,
        },
        "skops-tests/iris-sklearn-latest-hist_gradient_boosting-without-config": {
            "input": "iris-latest-input.json",
            "output": "iris-hist_gradient_boosting-latest-output.json",
            "has_config": False,
            "old_sklearn": False,
            "accepts_nan": True,
            "loads_on_new_sklearn": True,
        },
        "skops-tests/iris-sklearn-latest-hist_gradient_boosting-with-config": {
            "input": "iris-latest-input.json",
            "output": "iris-hist_gradient_boosting-latest-output.json",
            "has_config": True,
            "old_sklearn": False,
            "accepts_nan": True,
            "loads_on_new_sklearn": True,
        },
        "skops-tests/iris-sklearn-1.0-hist_gradient_boosting-without-config": {
            "input": "iris-1.0-input.json",
            "output": "iris-hist_gradient_boosting-1.0-output.json",
            "has_config": False,
            "old_sklearn": True,
            "accepts_nan": True,
            "loads_on_new_sklearn": False,
        },
        "skops-tests/iris-sklearn-1.0-hist_gradient_boosting-with-config": {
            "input": "iris-1.0-input.json",
            "output": "iris-hist_gradient_boosting-1.0-output.json",
            "has_config": True,
            "old_sklearn": True,
            "accepts_nan": True,
            "loads_on_new_sklearn": False,
        },
    },
    "tabular-regression": {
        "skops-tests/tabularregression-sklearn-latest-linear_regression-without-config": {
            "input": "tabularregression-latest-input.json",
            "output": "tabularregression-linear_regression-latest-output.json",
            "has_config": False,
            "old_sklearn": False,
            "accepts_nan": False,
            "loads_on_new_sklearn": True,
        },
        "skops-tests/tabularregression-sklearn-latest-linear_regression-with-config": {
            "input": "tabularregression-latest-input.json",
            "output": "tabularregression-linear_regression-latest-output.json",
            "has_config": True,
            "old_sklearn": False,
            "accepts_nan": False,
            "loads_on_new_sklearn": True,
        },
        "skops-tests/tabularregression-sklearn-1.0-linear_regression-without-config": {
            "input": "tabularregression-1.0-input.json",
            "output": "tabularregression-linear_regression-1.0-output.json",
            "has_config": False,
            "old_sklearn": True,
            "accepts_nan": False,
            "loads_on_new_sklearn": False,
        },
        "skops-tests/tabularregression-sklearn-1.0-linear_regression-with-config": {
            "input": "tabularregression-1.0-input.json",
            "output": "tabularregression-linear_regression-1.0-output.json",
            "has_config": True,
            "old_sklearn": True,
            "accepts_nan": False,
            "loads_on_new_sklearn": False,
        },
        "skops-tests/tabularregression-sklearn-latest-hist_gradient_boosting_regressor-without-config": {
            "input": "tabularregression-latest-input.json",
            "output": "tabularregression-hist_gradient_boosting_regressor-latest-output.json",
            "has_config": False,
            "old_sklearn": False,
            "accepts_nan": True,
            "loads_on_new_sklearn": True,
        },
        "skops-tests/tabularregression-sklearn-latest-hist_gradient_boosting_regressor-with-config": {
            "input": "tabularregression-latest-input.json",
            "output": "tabularregression-hist_gradient_boosting_regressor-latest-output.json",
            "has_config": True,
            "old_sklearn": False,
            "accepts_nan": True,
            "loads_on_new_sklearn": True,
        },
        "skops-tests/tabularregression-sklearn-1.0-hist_gradient_boosting_regressor-without-config": {
            "input": "tabularregression-1.0-input.json",
            "output": "tabularregression-hist_gradient_boosting_regressor-1.0-output.json",
            "has_config": False,
            "old_sklearn": True,
            "accepts_nan": True,
            "loads_on_new_sklearn": False,
        },
        "skops-tests/tabularregression-sklearn-1.0-hist_gradient_boosting_regressor-with-config": {
            "input": "tabularregression-1.0-input.json",
            "output": "tabularregression-hist_gradient_boosting_regressor-1.0-output.json",
            "has_config": True,
            "old_sklearn": True,
            "accepts_nan": True,
            "loads_on_new_sklearn": False,
        },
    },
    "text-classification": {
        "skops-tests/textclassification-sklearn-hist_gradient_boosting-latest-without-config": {
            "input": "textclassification-latest-input.json",
            "output": "textclassification-hist_gradient_boosting-latest-output.json",
            "has_config": False,
            "old_sklearn": False,
            "loads_on_new_sklearn": True,
        },
        "skops-tests/textclassification-sklearn-hist_gradient_boosting-latest-with-config": {
            "input": "textclassification-latest-input.json",
            "output": "textclassification-hist_gradient_boosting-latest-output.json",
            "has_config": True,
            "old_sklearn": False,
            "loads_on_new_sklearn": True,
        },
        "skops-tests/textclassification-sklearn-hist_gradient_boosting-1.0-without-config": {
            "input": "textclassification-1.0-input.json",
            "output": "textclassification-hist_gradient_boosting-1.0-output.json",
            "has_config": False,
            "old_sklearn": True,
            "loads_on_new_sklearn": False,
        },
        "skops-tests/textclassification-sklearn-hist_gradient_boosting-1.0-with-config": {
            "input": "textclassification-1.0-input.json",
            "output": "textclassification-hist_gradient_boosting-1.0-output.json",
            "has_config": True,
            "old_sklearn": True,
            "loads_on_new_sklearn": False,
        },
        "skops-tests/textclassification-sklearn-logistic_regression-latest-without-config": {
            "input": "textclassification-latest-input.json",
            "output": "textclassification-logistic_regression-latest-output.json",
            "has_config": False,
            "old_sklearn": False,
            "loads_on_new_sklearn": True,
        },
        "skops-tests/textclassification-sklearn-logistic_regression-latest-with-config": {
            "input": "textclassification-latest-input.json",
            "output": "textclassification-logistic_regression-latest-output.json",
            "has_config": True,
            "old_sklearn": False,
            "loads_on_new_sklearn": True,
        },
        "skops-tests/textclassification-sklearn-logistic_regression-1.0-without-config": {
            "input": "textclassification-1.0-input.json",
            "output": "textclassification-logistic_regression-1.0-output.json",
            "has_config": False,
            "old_sklearn": True,
            "loads_on_new_sklearn": False,
        },
        "skops-tests/textclassification-sklearn-logistic_regression-1.0-with-config": {
            "input": "textclassification-1.0-input.json",
            "output": "textclassification-logistic_regression-1.0-output.json",
            "has_config": True,
            "old_sklearn": True,
            "loads_on_new_sklearn": False,
        },
    },
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
