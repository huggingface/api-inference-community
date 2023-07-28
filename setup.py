from setuptools import setup


setup(
    name="api_inference_community",
    version="0.0.32",
    description="A package with helper tools to build an API Inference docker app for Hugging Face API inference using huggingface_hub",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="http://github.com/huggingface/api-inference-community",
    author="Nicolas Patry",
    author_email="nicolas@huggingface.co",
    license="MIT",
    packages=["api_inference_community"],
    python_requires=">=3.6.0",
    zip_safe=False,
    install_requires=list(line for line in open("requirements.txt", "r")),
    extras_require={
        "test": [
            "httpx>=0.18",
            "Pillow>=8.2",
            "httpx>=0.18",
            "torch>=1.9.0",
            "pytest>=6.2",
        ],
        "quality": ["black==22.3.0", "isort", "flake8", "mypy"],
    },
)
