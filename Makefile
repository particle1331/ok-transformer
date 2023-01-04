# See https://madewithml.com/courses/mlops/makefile/
.PHONY: docs
docs:
	jupyter-book build docs

rmdocs:
	rm -rf docs/_build
	make docs

clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".vscode" | xargs rm -rf
	rm -f .coverage
