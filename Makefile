.PHONY: $(MAKECMDGOALS)

build:
	uv run jupyter book start

rebuild:
	rm -rf _build
	$(MAKE) build

publish:
	uvx ghp-import -n -p -f _build/html

execute:
	uv run extras/execute.py --pattern $(pattern)

validate:
	uv run extras/validate.py

diff:
	uv run nbdiff-web $(old) $(new)
