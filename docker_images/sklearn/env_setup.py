"""This file sets up the environment for the model with the right required
packages installed.

The docker image for scikit-learn comes with mamba-forge which provides `mamba`
and `conda-forge` channel as the default channel.
"""
import json
import os
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = os.getenv("MODEL_ID")


def main():
    cached_folder = snapshot_download(repo_id=MODEL_ID)
    try:
        with open(Path(cached_folder) / "config.json") as f:
            # this is the default path for configuration of a scikit-learn
            # project. If the project is created using `skops`, it should have
            # this file.
            config = json.load(f)
        requirements = config["sklearn"]["environment"]
    except Exception as e:
        # If for whatever reason we fail to detect requirements of the project,
        # we install the latest scikit-learn.
        # TODO: we should log this, or warn or something.
        print(e)
        requirements = ["scikit-learn"]

    requirements += ["pandas", "uvicorn"]

    command = [
        "mamba",
        "create",
        "-y",
        "-q",
        "--name=api-inference-model-env",
    ] + requirements

    subprocess.run(
        ["mamba", "env", "remove", "-y", "-q", "--name=api-inference-model-env"]
    )
    print(command)
    subprocess.run(command, env=os.environ)


if __name__ == "__main__":
    main()
