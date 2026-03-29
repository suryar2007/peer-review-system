.PHONY: install test run

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

run:
	python main.py --paper $(PDF)
