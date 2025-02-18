"""Conftest module."""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
TESTS_DIR_PARENT = (THIS_DIR / "..").resolve()

sys.path.insert(0, str(TESTS_DIR_PARENT))

pytest_plugins = [
    "tests.fixtures.data_fixture",
    "tests.fixtures.basic_model_fixture",
    "tests.fixtures.custom_model_fixture",
    "tests.fixtures.model_serving_fixture",
]
