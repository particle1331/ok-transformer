.PHONY: docs
docs:
	jupyter-book build docs

pushall:
	git add .
	git commit -m ${m}
