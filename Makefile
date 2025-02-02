create-venv:
	uv venv -p 3.11 venv

install:
	uv pip install -r pyproject.toml --all-extras --link-mode=copy
	uv lock
	uv pip install -e . --link-mode=copy

lint:
	pre-commit run --all-files

clean:
	rm -rf __pycache__ dist

test:
	pytest

build:
	uv build

copy-whl-to-databricks:
	databricks fs cp ./dist/*.whl dbfs:/Volumes/mlops_dev/acikgozm/packages/credit_default-latest-py3-none-any.whl --overwrite
