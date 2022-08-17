#!/usr/bin/env python3

# This is an amost 1:1 copy of iris.py but uses regression instead of
# classification.

import json
import os
import pickle
import sys
import time
from pathlib import Path
from tempfile import mkdtemp, mkstemp

import pandas as pd
import sklearn
from huggingface_hub import HfApi
from huggingface_hub.utils._errors import RepositoryNotFoundError
from sklearn.datasets import make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skops import hub_utils


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

    # prevent AWS "503 Server Error: Slow Down for url" error
    time.sleep(10)
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


def get_tabularregression_data():
    # Create an artificial regression dataset
    n_features = 4
    X, y = make_regression(150, n_features=n_features, n_informative=2, random_state=42)
    columns = [f"col{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=columns), pd.Series(y)


def get_estimators(X, y):
    # yield estimator names and estimators to train and push to hub.

    # this is a pipeline with simple estimators which can be loaded across
    # different sklearn versions.
    yield "linear_regression", make_pipeline(StandardScaler(), LinearRegression()).fit(
        X, y
    )

    # this estimator cannot be loaded on 1.1 if it's stored using 1.0, but it
    # handles NaN input values which the previous pipeline cannot handle.
    yield "hist_gradient_boosting_regressor", HistGradientBoostingRegressor().fit(X, y)


def create_repos(est_name, est_instance, version):
    # given an estimator instance, it's name, and the version tag, train a
    # model and push to hub. Both with and w/o a config file.
    est = est_instance

    _, pkl_name = mkstemp(prefix="skops-", suffix=".pkl")
    with open(pkl_name, mode="bw") as f:
        pickle.dump(est, file=f)

    local_repo = mkdtemp(prefix="skops-")
    hub_utils.init(
        model=pkl_name,
        requirements=[f"scikit-learn={sklearn.__version__}"],
        dst=local_repo,
        task="tabular-regression",
        data=X_test,
    )

    push_repo(
        repo_name=f"tabularregression-sklearn-{version}-{est_name}-with-config",
        local_repo=local_repo,
    )

    # Now we remove the config file and push to a new repo
    os.remove(Path(local_repo) / "config.json")
    # The only valid file name for a model pickle file if no config.json is
    # available is `sklearn_model.joblib`, otherwise the backend will fail to
    # find the file.
    os.rename(Path(local_repo) / pkl_name, Path(local_repo) / "sklearn_model.joblib")

    push_repo(
        repo_name=f"tabularregression-sklearn-{version}-{est_name}-without-config",
        local_repo=local_repo,
    )

    # save model predictions, which are later used for tests
    with open(
        Path(__file__).parent
        / "samples"
        / f"tabularregression-{est_name}-{version}-output.json",
        "w",
    ) as f:
        output = [float(x) for x in est.predict(X_test.iloc[:10, :])]
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    version = sys.argv[1]

    X, y = get_tabularregression_data()
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    for est_name, est_instance in get_estimators(X_train, y_train):
        create_repos(est_name, est_instance, version)

    # take the first 10 rows as a sample input to the model.
    sample = X_test.head(10).to_dict(orient="list")

    # save model input, which are later used for tests
    payload = {"data": sample}
    with open(
        Path(__file__).parent / "samples" / f"tabularregression-{version}-input.json",
        "w",
    ) as f:
        json.dump(payload, f, indent=2)
