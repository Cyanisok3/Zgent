.PHONY: lint test integration-test docs benchmark-ci verify-s0

lint:
	uv run ruff check src tests scripts
	uv run mypy src

test:
	uv run pytest tests/unit -v

integration-test:
	uv run pytest tests/integration -v

docs:
	uv run python scripts/gen_protocol_doc.py

benchmark-ci:
	uv run pytest tests/unit/test_evidence_benchmark.py -v

verify-s0:
	uv sync --frozen
	uv run ruff check src tests scripts
	uv run mypy src
	uv run pytest tests/unit -v
	uv run pytest tests/integration -k ping -v
	uv run python scripts/gen_protocol_doc.py --check
