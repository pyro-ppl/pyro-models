.PHONY: all lint license FORCE

all: lint

lint: FORCE
	python scripts/update_headers.py --check

license: FORCE
	python scripts/update_headers.py

FORCE:
