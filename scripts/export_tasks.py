"""Exports a library -> supported tasks mapping in JSON format. 

This script
- parses the source code of a library's app/main.py and extracts the AST
- finds the ALLOWED_TASKS variable and get all the keys.
- prints the library name as well as its tasks in JSON format.

Note that the transformer library is not included in the output 
as we can assume it supports all tasks. This is done as
the transformers API codebase is not in this repository.
"""

import ast
import collections
import os
import pathlib
import json


lib_to_task_map = collections.defaultdict(list)


def _extract_tasks(library_name, variable_name, value):
    """Extract supported tasks of the library.

    Args:
        library_name: The name of the library (e.g. paddlenlp)
        variable_name: The name of the Python variable (e.g. ALLOWED_TASKS)
        value: The AST of the variable's Python value.
    """
    if variable_name == "ALLOWED_TASKS":
        if isinstance(value, ast.Dict):
            for key in value.keys:
                lib_to_task_map[library_name].append(key.value)


def traverse_global_assignments(library_name, file_content, handler):
    """Traverse all global assignments and apply handler on each of them.

    Args:
        library_name: The name of the library (e.g. paddlenlp)
        file_content: The content of app/main.py file in string.
        handler: A callback that processes the AST.
    """
    for element in ast.parse(file_content).body:
        # Typical case, e.g. TARGET_ID: Type = VALUE
        if isinstance(element, ast.AnnAssign):
            handler(library_name, element.target.id, element.value)
        # Just in case user omitted the type annotation
        # Unpacking and multi-variable assignment is rare so not handled
        # e.g. TARGET_ID = VALUE
        elif isinstance(element, ast.Assign):
            target = element.targets[0]
            if isinstance(target, ast.Name):
                handler(library_name, target.id, element.value)


if __name__ == "__main__":
    root = pathlib.Path(__file__).parent.parent.resolve()
    libs = os.listdir(root / "docker_images")
    libs.remove("common")

    for lib in libs:
        with open(root / "docker_images" / lib / "app/main.py") as f:
            content = f.read()
            traverse_global_assignments(lib, content, _extract_tasks)

    output = json.dumps(lib_to_task_map, sort_keys=True, indent=4)
    print(output)
