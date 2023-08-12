# See https://madewithml.com/courses/mlops/makefile/
.PHONY: docs
dev:
	tox -e build

docs:
	rm -rf docs/_build
	tox -e build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	rm -rf htmlcov/
	rm -f .coverage
	find . | grep ".egg" | xargs rm -rf
	find . | grep ".vscode" | xargs rm -rf
	find . | grep ".egg-info" | xargs rm -rf
	find . | grep ".DS_Store" | xargs rm
	find . | grep ".pytest_cache" | xargs rm -rf
	find . | grep ".ipynb_checkpoints" | xargs rm -rf
	find . | grep "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf

make create:
	python extras/init.py $(filename) $(title)
