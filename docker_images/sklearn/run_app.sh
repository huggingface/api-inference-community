#!/bin/bash --login
# This file creates an environment with all required dependencies for the given
# model, and then runs the start command.

# This makes it easy to see in logs what exactly is being run.
set -xe

get_requirements() {
    requirements="pandas uvicorn gunicorn api-inference-community"
    # this next command is needed to run the while loop in the same process and
    # therefore modify the same $requirements variable. Otherwise the loop would be
    # a separate process and the variable wouldn't be accessible from this parent
    # process.
    shopt -s lastpipe
    jq '.sklearn.environment' /tmp/config.json | jq '.[]' | while read r; do
        requirements+=" $r"
    done

    # not sure why these are required. But if they're not here, the string passed
    # to micromamba is kinda not parsable by it.
    requirements=$(echo "$requirements" | sed "s/'//g")
    requirements=$(echo "$requirements" | sed "s/\"//g")
    echo $requirements
}

# We download only the config file and use `jq` to extract the requirements. If
# the download fails, we use a default set of dependencies. We need to capture
# the output of `curl` here so that if it fails, it doesn't make the whole
# script to exit, which it would do due to the -e flag we've set above the
# script.
response="$(curl https://huggingface.co/$MODEL_ID/raw/main/config.json -f --output /tmp/config.json)" || response=$?
if [ -z $response ]; then
    requirements=$(get_requirements)
else
    # if the curl command is not successful, we use a default set of
    # dependencies, and use the latest scikit-learn version. This is to allow
    # users for a basic usage if they haven't put the config.json file in their
    # repository.
    requirements="pandas uvicorn gunicorn api-inference-community scikit-learn"
fi

micromamba create -c conda-forge -y -q --name=api-inference-model-env $requirements

micromamba activate api-inference-model-env

# start.sh file is not in our repo, rather taken from the
# `uvicorn-gunicorn-docker` repo. You can check the Dockerfile to see where
# exactly it is coming from.
/start.sh
