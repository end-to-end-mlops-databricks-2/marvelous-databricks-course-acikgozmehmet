"""Unit tests for the custom model"""
import pytest

from hotel_reservations.custom_model import load_model
from tests.conftest import TESTS_DIR_PARENT


def test_load_model() -> None:
    filepath = (TESTS_DIR_PARENT / "tests" /"test_data" / "lightgbm-pipeline-model" / "model.pkl").resolve()
    print(f'{filepath =}')
    model = load_model(filepath.as_posix())
    assert model is not None
