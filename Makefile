.PHONY: install run clean test

install:
	uv sync --group dev

run:
	PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python main.py

clean:
	rm -rf .venv __pycache__ **/__pycache__ *.pyc **/*.pyc
	rm -rf .pytest_cache build dist *.egg-info

test:
	uv run pytest tests/

dev:
	uv run ipython

