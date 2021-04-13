
This repositories enable third-party libraries integrated with [huggingface_hub](https://github.com/huggingface/huggingface_hub/) to create
their own docker so that the widgets on the hub can work as the `transformers` one do.

The hardware to run the API will be provided by Hugging Face for now.

The `common` folder is intended to be a starter point for all new libs that 
want to be integrated.

### Adding a new container from a new lib.


1. Copy the `common` folder into your library's name `example`.
2. Edit:
    - `example/requirements.txt`
    - `example/app/main.py`
    - `example/app/pipelines/{task_name}.py` 
    to implement the desired functionnality. All required code is marked with `IMPLEMENT_THIS` markup.
3. Feel free to customize anything required by your lib everywhere you want. The only real requirements, are to honor the HTTP endpoints, in the same fashion as the `common` folder for all your supported tasks.
4. Edit `example/tests/test_api.py` to add TESTABLE_MODELS.
5. Pass the test suite `pytest -sv --rootdir example/ example/`
6. Enjoy !


### Available tasks

- [Automatic speech recognition]: Input is a file, output is a dict of understood words being said within the file
- [Text generation]: Input is a text, output is a dict of generated text
- [Image recognition]: Input is a text, output is a dict of generated text


