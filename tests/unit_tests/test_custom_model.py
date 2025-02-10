"""Unit tests for the custom model."""

from hotel_reservations.custom_model import load_model
from tests.consts import PROJECT_DIR


def test_load_model() -> None:
    """Test the loading of a LightGBM pipeline model.

    Ensures that the model is successfully loaded from the specified file path.
    """
    filepath = (PROJECT_DIR / "tests" / "test_data" / "lightgbm-pipeline-model" / "model.pkl").resolve()
    model = load_model(filepath.as_posix())
    assert model is not None
