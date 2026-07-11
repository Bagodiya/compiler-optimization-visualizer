"""Tests for the diff command skeleton.

There's no real diffing wired up yet, so for now we just make sure the
command exists, validates its input file, and reports the levels it will
compare. The actual diff behaviour gets its own tests once it lands.
"""

from pathlib import Path

from typer.testing import CliRunner

from compopt.cli import app

runner = CliRunner()


def test_diff_reports_levels(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    result = runner.invoke(app, ["diff", str(src)])
    assert result.exit_code == 0
    # the placeholder should name the two levels it's going to compare
    assert "-O0" in result.stdout
    assert "-O2" in result.stdout


def test_diff_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.c"

    result = runner.invoke(app, ["diff", str(missing)])
    assert result.exit_code == 1


def test_diff_directory_is_rejected(tmp_path: Path) -> None:
    # a directory isn't a source file, so this should fail like a missing one
    result = runner.invoke(app, ["diff", str(tmp_path)])
    assert result.exit_code == 1
