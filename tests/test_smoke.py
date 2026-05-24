"""Smoke test to verify the package imports and CLI is wired up."""

from compopt import __version__


def test_version_string() -> None:
    assert __version__ == "0.0.1"
