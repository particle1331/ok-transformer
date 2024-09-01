# See https://madewithml.com/courses/mlops/makefile/
docs:
	tox -e build

.PHONY: docs
rdocs:
	rm -rf docs/_build
	tox -e build

clean:
	find . | grep ".egg" | xargs rm -rf
	find . | grep ".vscode" | xargs rm -rf
	find . | grep ".egg-info" | xargs rm -rf
	find . | grep ".DS_Store" | xargs rm
	find . | grep ".pytest_cache" | xargs rm -rf
	find . | grep ".ipynb_checkpoints" | xargs rm -rf
	find . | grep "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf

create:
	python extras/init.py "$(filename)" "$(title)"

ruff-check:
	- pdm run ruff check $(path)
	- pdm run ruff check --select I --diff $(path)
	- pdm run ruff format --diff $(path)

ruff-format:
	pdm run ruff check --select I --fix $(path)
	pdm run ruff format $(path)

execute:
	pdm run extras/run.py --folder $(folder)
