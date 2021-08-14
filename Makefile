.PHONY: docs
docs:
	jupyter-book build docs

git:
	git add .
	git commit -m "${m}"
	git push
