#!/bin/bash
# This file is needed to first activate the environment created for this
# project and then run the web server. The environment is created by the
# `env_setup.py` script.

conda init bash
source ~/.bashrc
conda activate api-inference-model-env
pip install api-inference-community
python app/run_app.py
