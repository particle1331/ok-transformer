.PHONY: docs
docs:
	rm -rf docs/_build
	jupyter-book build docs

.PHONY: clean
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

.ONESHELL: docs
deploy:
	ghp-import -n -p -f docs/_build/html

.ONESHELL:
commit-deploy: commit deploy
