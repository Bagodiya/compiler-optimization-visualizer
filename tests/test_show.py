"""Tests for the show command skeleton."""

from pathlib import Path

from typer.testing import CliRunner

from compopt.cli import app

runner = CliRunner()


def test_show_existing_file(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int main(void) { return 0; }\n")

    result = runner.invoke(app, ["show", str(src)])
    assert result.exit_code == 0
    assert "would show" in result.stdout
    assert str(src) in result.stdout


def test_show_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.c"

    result = runner.invoke(app, ["show", str(missing)])
    assert result.exit_code == 1


def test_show_directory_is_rejected(tmp_path: Path) -> None:
    # passing a directory instead of a file should fail too
    result = runner.invoke(app, ["show", str(tmp_path)])
    assert result.exit_code == 1
