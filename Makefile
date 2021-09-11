.PHONY: docs
docs:
	jupyter-book build docs

.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

.ONESHELL:
git: docs clean
	git add .
	git commit -m "${m}"
	git push
	ghp-import -n -p -f docs/_build/html