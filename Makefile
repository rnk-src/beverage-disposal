# One command for every gate. CI runs exactly this.
.PHONY: check
check:
	uv run --frozen ruff format --check .
	uv run --frozen ruff check .
	uv run --frozen mypy
	uv run --frozen pytest --ignore=tests/gazebo
