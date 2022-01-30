.PHONY: docs
docs:
	jupyter-book build docs

docs-rm:
	rm -rf docs/_build
	make docs

.ONESHELL: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

.ONESHELL:
commit: clean
	git add .
	git commit -m "${m}"
	git push

.ONESHELL:
push: docs-rm
	ghp-import -n -p -f docs/_build/html
	git status

.ONESHELL:
deploy: commit docs-rm
	ghp-import -n -p -f docs/_build/html
	git status