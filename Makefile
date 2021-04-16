.PHONY: quality style


check_dirs := .



quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

style:
	black $(check_dirs)
	isort $(check_dirs)


test:
	pytest -sv tests/

