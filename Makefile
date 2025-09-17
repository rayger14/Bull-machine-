
.PHONY: venv install fmt lint type test smoke prepush clean

venv:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip

install:
	. .venv/bin/activate && pip install -U numpy pandas pydantic==2.* pytest pytest-cov mypy ruff

fmt:
	. .venv/bin/activate && ruff format .

lint:
	. .venv/bin/activate && ruff check . --fix

type:
	. .venv/bin/activate && mypy bull_machine --ignore-missing-imports

test:
	. .venv/bin/activate && pytest -q --disable-warnings --maxfail=1

smoke:
	. .venv/bin/activate && \
	python - <<'PY'\nimport json; cfg=json.load(open('bull_machine/config/config_v1_2_1.json')); print('Config OK:', cfg['version'])\nPY

prepush: venv install fmt lint type test smoke
	@echo "✅ All checks passed - ready to push!"

clean:
	rm -rf .venv __pycache__ .pytest_cache .coverage .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
