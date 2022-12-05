import ast
import os
import pathlib
import json

lib_to_task_map = {}


def extract_tasks(library_name, variable_name, value):
    if variable_name == "ALLOWED_TASKS":
        if isinstance(value, ast.Dict):
            for key in value.keys:
                lib_to_task_map.setdefault(library_name, []).append(key.value)


def traverse_global_assignments(library_name, file_content, handler):
    for element in ast.parse(file_content).body:
        # Typical case
        if isinstance(element, ast.AnnAssign):
            handler(library_name, element.target.id, element.value)
        # Just in case user omitted the type annotation
        # Unpacking and multi-variable assignment is rare so not handled
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
            traverse_global_assignments(lib, content, extract_tasks)

    output = json.dumps(lib_to_task_map, sort_keys=True, indent=4)
    print(output)
