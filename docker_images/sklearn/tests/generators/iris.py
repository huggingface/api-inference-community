#!/usr/bin/env python

import json
import os
import pickle
import sys
from pathlib import Path
from tempfile import mkdtemp, mkstemp

import sklearn
from huggingface_hub import HfApi
from requests import HTTPError
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skops import hub_utils


def push_repo(repo_name, local_repo):
    # this token should be allowed to push to the skops-tests org.
    token = os.environ["SKOPS_TESTS_TOKEN"]
    repo_id = f"skops-tests/{repo_name}"

    answer = input(f"Do you want to publish this model under {repo_id}? [y/N] ")
    if answer != "y":
        return

    client = HfApi()

    try:
        client.model_info(repo_id=repo_id, token=token)
    except HTTPError:
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


if __name__ == "__main__":
    version = sys.argv[1]

    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    est = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_train, y_train)

    _, pkl_name = mkstemp(prefix="skops-", suffix=".pkl")
    with open(pkl_name, mode="bw") as f:
        pickle.dump(est, file=f)

    local_repo = mkdtemp(prefix="skops-")
    hub_utils.init(
        model=pkl_name,
        requirements=[f"scikit-learn={sklearn.__version__}"],
        dst=local_repo,
        task="tabular-classification",
        data=X,
    )

    push_repo(
        repo_name=f"iris-sklearn-{version}-with-config",
        local_repo=local_repo,
    )

    # Now we remove the config file and push to a new repo
    os.remove(Path(local_repo) / "config.json")

    push_repo(
        repo_name=f"iris-sklearn-{version}-without-config",
        local_repo=local_repo,
    )

    # generate and save sample input data
    sample = dict()
    for column in X.columns:
        sample[column] = X[column][:10].to_list()

    payload = {"data": sample}
    with open(
        Path(__file__).parent / "samples" / f"iris-{version}-input.json", "w"
    ) as f:
        json.dump(payload, f, indent=2)

    with open(
        Path(__file__).parent / "samples" / f"iris-{version}-output.json", "w"
    ) as f:
        output = [int(x) for x in est.predict(X_test.iloc[:10, :])]
        json.dump(output, f, indent=2)
