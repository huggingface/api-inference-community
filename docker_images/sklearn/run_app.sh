#!/bin/bash --login
# This file creates an environment with all required dependencies for the given
# model, and then runs the start command.

# This makes it easy to see in logs what exactly is being run.
set -xe

# We download only the config file and use `jq` to extract the requirements.
curl https://huggingface.co/$MODEL_ID/raw/main/config.json --output /tmp/config.json 

requirements="pandas uvicorn gunicorn"
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

micromamba create -c conda-forge -y -q --name=api-inference-model-env $requirements

micromamba activate api-inference-model-env

pip install api-inference-community

# This file is not in our repos, rather taken from the
# `uvicorn-gunicorn-docker` repos. You can check the Dockerfile to see where
# exactly they are coming from.
/start.sh
