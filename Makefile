# See https://madewithml.com/courses/mlops/makefile/
.PHONY: docs
docs:
	jupyter-book build docs

docsrm:
	rm -rf docs/_build
	make docs

clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".vscode" | xargs rm -rf
	rm -f .coverage

.ONESHELL:
commit: clean
	git add .
	git commit -m "${m}"
	git push

push: docsrm
	git pull
	ghp-import -n -p -f docs/_build/html
	git status

deploy: commit docsrm
	ghp-import -n -p -f docs/_build/html
	git status
