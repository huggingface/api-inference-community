## Tests

The tests require certain repositories with certain requirements on the hub.

You can make sure those repos are up to date by running the
`docker_images/sklearn/tests/generators/run.sh` script. The script creates
required conda environments and runs scripts inside those environments. You
should also give it a valid token with access to the `skops-tests` org:

```
SKOPS_TESTS_TOKEN=your_secret_token path/to/run.sh
```

You also need the latest `scikit-learn` installed in your current environment
in order to run the tests with:

```
pytest -sv --rootdir docker_images/sklearn/ docker_images/sklearn/
```
