.PHONY: install lint typecheck test download-data train evaluate tune promote validate-bundle serve dashboard

install:
	uv sync --all-groups

lint:
	uv run ruff check .

typecheck:
	uv run mypy src scripts tests

test:
	uv run pytest

download-data:
	uv run python scripts/download_data.py

train:
	uv run python scripts/train.py

evaluate:
	uv run python scripts/evaluate.py

tune:
	uv run python scripts/tune.py

promote:
	uv run python scripts/promote.py

validate-bundle:
	uv run python scripts/validate_bundle.py

serve:
	uv run python scripts/serve.py

dashboard:
	uv run streamlit run streamlit_app.py
