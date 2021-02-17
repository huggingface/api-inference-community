.PHONY: quality style test


check_dirs := tests main.py


quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	# flake8 $(check_dirs)

style:
	black $(check_dirs)
	isort $(check_dirs)

test:
	pytest -sv ./tests/ -W ignore::DeprecationWarning

