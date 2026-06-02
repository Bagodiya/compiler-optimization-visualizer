"""Tests for the command-line interface."""

from typer.testing import CliRunner

from compopt import __version__
from compopt.cli import app

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"compopt {__version__}" in result.stdout


def test_help_mentions_optimization() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "optimization" in result.stdout.lower()


def test_no_args_shows_help() -> None:
    # no_args_is_help means a bare invocation should still exit cleanly
    result = runner.invoke(app, [])
    assert "Usage" in result.stdout
