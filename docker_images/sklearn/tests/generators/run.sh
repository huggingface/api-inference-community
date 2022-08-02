#!/usr/bin/env bash

# uncomment to enable debugging
# set -xe

# have to do this since can't do mamba run, and need bash functions to call
# activate
source $(mamba info -q --base)/etc/profile.d/conda.sh 
source $(mamba info -q --base)/etc/profile.d/mamba.sh 

# mamba env update --file sklearn-1.0.yml
# mamba env update --file sklearn-latest.yml

# not doing mamba run ... since it just wouldn't work and would use system's
# python
mamba activate api-inference-community-test-generator-sklearn-1-0
python iris.py 1.0
mamba deactivate

mamba activate api-inference-community-test-generator-sklearn-latest
python iris.py latest
mamba deactivate
