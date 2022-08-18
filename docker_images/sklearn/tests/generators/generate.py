#!/usr/bin/env python3

"""Generate artefacts used for testing

Don't run this script directly but use `run.sh` instead.

For the given sklearn version, train models for different task types, upload
them with and without config to HF Hub, and store their input and predictions
locally (and in the GH repo).

These artefacts will be used for unit testing the sklearn integration.

"""

import json
import os
import pickle
import sys
import time
from pathlib import Path
from tempfile import mkdtemp, mkstemp

import sklearn
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skops import hub_utils


SLEEP_BETWEEN_PUSHES = 10


def push_repo(repo_name, local_repo):
    # this token should be allowed to push to the skops-tests org.
    token = os.environ["SKOPS_TESTS_TOKEN"]
    repo_id = f"skops-tests/{repo_name}"

    print(f"Pushing {repo_id}")

    client = HfApi()
    try:
        client.delete_repo(repo_id, token=token)
    except RepositoryNotFoundError:
        # repo does not exist yet
        pass
    client.create_repo(repo_id=repo_id, token=token, repo_type="model")

    client.upload_folder(
        repo_id=repo_id,
        path_in_repo=".",
        folder_path=local_repo,
        commit_message="pushing files to the repo from test generator!",
        commit_description=None,
        token=token,
        repo_type=None,
        revision=None,
        create_pr=False,
    )
    # prevent AWS "503 Server Error: Slow Down for url" error
    time.sleep(SLEEP_BETWEEN_PUSHES)


def get_tabular_classifiers():
    # yield classifier names and estimators to train and push to hub.

    # this is a pipeline with simple estimators which can be loaded across
    # different sklearn versions.
    yield "logistic_regression", make_pipeline(StandardScaler(), LogisticRegression())

    # this estimator cannot be loaded on 1.1 if it's stored using 1.0, but it
    # handles NaN input values which the previous pipeline cannot handle.
    yield "hist_gradient_boosting", HistGradientBoostingClassifier()


def get_tabular_regressors():
    # yield regressor names and estimators to train and push to hub.

    # this is a pipeline with simple estimators which can be loaded across
    # different sklearn versions.
    yield "linear_regression", make_pipeline(StandardScaler(), LinearRegression())

    # this estimator cannot be loaded on 1.1 if it's stored using 1.0, but it
    # handles NaN input values which the previous pipeline cannot handle.
    yield "hist_gradient_boosting_regressor", HistGradientBoostingRegressor()


def create_repos(est_name, task_name, est, sample, version):
    # given trained estimator instance, it's name, and the version tag, push to
    # hub once with and once without a config file.

    # initialize repo
    _, pkl_name = mkstemp(prefix="skops-", suffix=".pkl")
    with open(pkl_name, mode="bw") as f:
        pickle.dump(est, file=f)

    local_repo = mkdtemp(prefix="skops-")
    hub_utils.init(
        model=pkl_name,
        requirements=[f"scikit-learn={sklearn.__version__}"],
        dst=local_repo,
        task=task_name,
        data=sample,
    )

    # push WITH config
    repo_name = REPO_NAMES[task_name].format(
        version=version, est_name=est_name, w_or_wo="with"
    )
    push_repo(repo_name=repo_name, local_repo=local_repo)

    # push WIHTOUT CONFIG
    repo_name = REPO_NAMES[task_name].format(
        version=version, est_name=est_name, w_or_wo="without"
    )

    # Now we remove the config file and push to a new repo
    os.remove(Path(local_repo) / "config.json")
    # The only valid file name for a model pickle file if no config.json is
    # available is `sklearn_model.joblib`, otherwise the backend will fail to
    # find the file.
    os.rename(Path(local_repo) / pkl_name, Path(local_repo) / "sklearn_model.joblib")

    push_repo(
        repo_name=repo_name,
        local_repo=local_repo,
    )


def save_sample(sample, filename):
    payload = {"data": sample.to_dict(orient="list")}
    with open(Path(__file__).parent / "samples" / filename, "w") as f:
        json.dump(payload, f, indent=2)


def predict_tabular_classifier(est, sample, filename):
    output = [int(x) for x in est.predict(sample)]
    with open(Path(__file__).parent / "samples" / filename, "w") as f:
        json.dump(output, f, indent=2)


def predict_tabular_regressor(est, sample, filename):
    output = [float(x) for x in est.predict(sample)]
    with open(Path(__file__).parent / "samples" / filename, "w") as f:
        json.dump(output, f, indent=2)


#############
# CONSTANTS #
#############

TASKS = [
    "tabular-classification",
    "tabular-regression",
]
DATA = {
    "tabular-classification": load_iris(return_X_y=True, as_frame=True),
    "tabular-regression": load_diabetes(return_X_y=True, as_frame=True),
}
MODELS = {
    "tabular-classification": get_tabular_classifiers(),
    "tabular-regression": get_tabular_regressors(),
}
INPUT_NAMES = {
    "tabular-classification": "iris-{version}-input.json",
    "tabular-regression": "tabularregression-{version}-input.json",
}
OUTPUT_NAMES = {
    "tabular-classification": "iris-{est_name}-{version}-output.json",
    "tabular-regression": "tabularregression-{est_name}-{version}-output.json",
}
REPO_NAMES = {
    "tabular-classification": "iris-sklearn-{version}-{est_name}-{w_or_wo}-config",
    "tabular-regression": "tabularregression-sklearn-{version}-{est_name}-{w_or_wo}-config",
}
PREDICT_FUNCTIONS = {
    "tabular-classification": predict_tabular_classifier,
    "tabular-regression": predict_tabular_regressor,
}


def main(version):
    for task in TASKS:
        print(f"Creating data for task '{task}' and version '{version}'")
        X, y = DATA[task]
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        sample = X_test.head(10)

        # save model input, which are later used for tests
        input_name = INPUT_NAMES[task].format(version=version)
        save_sample(sample, input_name)

        for est_name, model in MODELS[task]:
            model.fit(X_train, y_train)
            create_repos(
                est_name=est_name,
                task_name=task,
                est=model,
                sample=sample,
                version=version,
            )

            # save model predictions, which are later used for tests
            output_name = OUTPUT_NAMES[task].format(est_name=est_name, version=version)
            predict = PREDICT_FUNCTIONS[task]
            predict(model, sample, output_name)


if __name__ == "__main__":
    sklearn_version = sys.argv[1]
    main(sklearn_version)
