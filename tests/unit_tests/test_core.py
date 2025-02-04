"""Test the core module."""

from hotel_reservations import __version__


# @pytest.mark.skip(reason="wip")
def test_package_version_should_have_value() -> None:
    """Test that the package version is set.

    :return: None
    """
    assert __version__
