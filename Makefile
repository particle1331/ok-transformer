.PHONY: docs
docs:
	jupyter-book build docs

.ONESHELL:
git:
	git add .
	git commit -m "${m}"
	git push
