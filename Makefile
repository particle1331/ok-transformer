.PHONY: docs
docs:
	tox -e build

rmdocs:
	rm -rf docs/_build
	tox -e build

clean:
	find . | grep ".egg" | xargs rm -rf
	find . | grep ".vscode" | xargs rm -rf
	find . | grep ".egg-info" | xargs rm -rf
	find . | grep ".DS_Store" | xargs rm
	find . | grep ".pytest_cache" | xargs rm -rf
	find . | grep ".ipynb_checkpoints" | xargs rm -rf
	find . | grep "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf

create:
	python extras/init.py "$(filename)" "$(title)"

execute:
	pdm run extras/run.py --pattern $(pattern)

validate:
	pdm run extras/validate.py
