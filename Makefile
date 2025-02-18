create-venv:
	uv venv -p 3.11 venv

install:
	uv pip install -r pyproject.toml --all-extras --link-mode=copy  --no-cache-dir
	uv lock
	uv pip install -e . --link-mode=copy  --no-cache-dir

lint:
	pre-commit run --all-files

clean:
	rm -rf __pycache__ dist

test:
	pytest

build:
	uv build

copy-whl-to-dbx:
	databricks fs cp ./dist/*.whl dbfs:/Volumes/mlops_dev/acikgozm/packages/hotel_reservations-latest-py3-none-any.whl --overwrite

deploy: clean build copy-whl-to-dbx
