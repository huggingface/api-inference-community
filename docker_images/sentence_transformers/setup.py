from setuptools import setup, find_packages
import os


VERSION = "0.0.1"

def parse_requirements(filename):
    """ Load requirements from a pip requirements file """
    with open(filename, 'r') as f:
        return f.read().splitlines()

# Specify the path to the requirements.txt file
requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')

print("Discovered packages:", find_packages())


setup(
    name='hf_api_sentence_transformers',
    version=VERSION,
    packages=find_packages(),
    install_requires=parse_requirements(requirements_path)
)