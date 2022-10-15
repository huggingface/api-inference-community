## Tests

### Test setup

The tests require certain repositories with certain requirements to exist on HF
Hub and certain output files to be created.

You can make sure those repos and files are up to date by running the
`docker_images/sklearn/tests/generators/run.sh` script. The script creates
required conda environments, updates them if necessary, and runs scripts inside
those environments. You should also give it a valid token with access to the
`skops-tests` org:

```bash
# from the project root
SKOPS_TESTS_TOKEN=your_secret_token docker_images/sklearn/tests/generators/run.sh
```

This script needs to be run _only once_ when you first start developing, or each
time a new scikit-learn version is released.

The created model repositories are also used for common tests of this package,
see `tests/test_dockers.py` > `test_sklearn`.

Note that a working [mamba
installation](https://mamba.readthedocs.io/en/latest/installation.html) is
required for this step

### Test environment

Create a new Python environment and install the test dependencies:

```bash
# with pip
python -m pip install -r docker_images/sklearn/requirements.txt
# with conda/mamba
conda install --file docker_images/sklearn/requirements.txt
```

### Running the tests

From within the Python environment, run:

```
pytest -sv --rootdir docker_images/sklearn/ docker_images/sklearn/
```

You will see many tests being skipped. If the message is "Skipping test because
requirements are not met.", it means that the test was intended to be skipped,
so you don't need to do anything about it. When adding a new test, make sure
that at least one of the parametrized settings is not skipped for that test.

### Adding a new task

When adding tests for a new task, certain artifacts like HF Hub repositories,
model inputs, and model outputs need to be generated first using the `run.sh`
script, as explained above. For the new task, those have to be implemented
first. For this, visit `docker_images/sklearn/tests/generators/generate.py` and
extend the script to include the new task. Most notably, visit the "CONSTANTS"
section and extend the constants defined there to include your task. This will
make it obvious which extra functions you need to write.
