.PHONY: docs
docs:
	jupyter-book build docs

.ONESHELL:
git: docs
	git add .
	git commit -m "${m}"
	git push
