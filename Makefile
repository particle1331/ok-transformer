.PHONY: $(MAKECMDGOALS)

build:
	uv run jupyter book start

rebuild:
	rm -rf _build
	$(MAKE) build

publish:
	uvx ghp-import -n -p -f _build/html

execute:
	uv run python extras/execute.py --pattern $(pattern)

validate:
	uv run python extras/validate.py

init:
	uv sync
	cp -r extras/ipython/* ~/.ipython/profile_default/startup/
