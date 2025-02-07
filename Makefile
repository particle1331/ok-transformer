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

# create:
# 	python extras/init.py "$(filename)" "$(title)"

# clean:
# 	find . | grep ".egg" | xargs rm -rf
# 	find . | grep ".vscode" | xargs rm -rf
# 	find . | grep ".egg-info" | xargs rm -rf
# 	find . | grep ".DS_Store" | xargs rm
# 	find . | grep ".pytest_cache" | xargs rm -rf
# 	find . | grep ".ipynb_checkpoints" | xargs rm -rf
# 	find . | grep "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
