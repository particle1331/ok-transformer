.PHONY: $(MAKECMDGOALS)

build:
	uvx --with "sphinx-rtd-theme" --with "jupyter-book==1.0.2" jupyter-book build docs

rebuild:
	rm -rf docs/_build
	$(MAKE) build

publish:
	uvx ghp-import -n -p -f docs/_build/html

execute:
	uv run python extras/run.py --pattern $(pattern)

validate:
	uv run python extras/validate.py
