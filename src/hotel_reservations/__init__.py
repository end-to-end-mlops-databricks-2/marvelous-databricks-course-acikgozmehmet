"""hotel_reservations package."""

import importlib.metadata
import importlib.resources


def get_version() -> str:
    """Retrieve the version of the package.

    This function attempts to read the version from the installed package,
    and if not found, reads it from a version.txt file.

    :return: The version string of the package.
    """
    try:
        # Try to read from the installed package
        return importlib.metadata.version("hotel_reservations")
    except importlib.metadata.PackageNotFoundError:
        # If not installed, read from the version.txt file
        with (
            importlib.resources.files("hotel_reservations")
            .joinpath("../../version.txt")
            .open("r", encoding="utf-8") as file
        ):
            return file.read().strip()


__version__ = get_version()
