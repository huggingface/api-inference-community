## Tests

The tests require certain repositories with certain requirements on the hub.

You can make sure those repos are up to date by running the
`docker_images/sklearn/tests/generators/run.sh` script. The script creates
required conda environments and runs scripts inside those environments.

You also need the latest `scikit-learn` installed in your current environment
in order to run the tests with:

```
pytest -sv --rootdir docker_images/sklearn/ docker_images/sklearn/
```
