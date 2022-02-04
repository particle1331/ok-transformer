# Since docs is unavailable (already used for a folder name), we have to use .PHONY
.PHONY: docs
docs:
	jupyter-book build docs

docs-rm:
	rm -rf docs/_build
	make docs

# Default: Each line in a recipe for a rule will execute in a separate sub-shell.
# Using .ONESHELL executes all steps in a single shell.
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

push: docs-rm
	ghp-import -n -p -f docs/_build/html
	git status

deploy: commit docs-rm
	ghp-import -n -p -f docs/_build/html
	git status