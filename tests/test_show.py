"""Tests for the show command."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from compopt.cli import app
from compopt.compilers import find_compilers

runner = CliRunner()

# most of these need a real compiler to produce assembly
needs_compiler = pytest.mark.skipif(
    not find_compilers(), reason="no gcc or clang available"
)


@needs_compiler
def test_show_prints_asm(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    result = runner.invoke(app, ["show", str(src)])
    assert result.exit_code == 0
    # the function name ends up as a label in the generated asm
    assert "add" in result.stdout


@needs_compiler
def test_show_func_picks_named_function(tmp_path: Path) -> None:
    src = tmp_path / "two.c"
    src.write_text(
        "int add(int a, int b) { return a + b; }\n"
        "int sub(int a, int b) { return a - b; }\n"
    )

    result = runner.invoke(app, ["show", str(src), "--func", "sub"])
    assert result.exit_code == 0
    assert "sub:" in result.stdout
    # asking for sub shouldn't drag add along with it
    assert "add:" not in result.stdout


@needs_compiler
def test_show_unknown_func_errors(tmp_path: Path) -> None:
    src = tmp_path / "two.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    result = runner.invoke(app, ["show", str(src), "--func", "nope"])
    assert result.exit_code == 1
    # the message should point the user at what's actually there
    assert "add" in result.stdout + result.stderr


@needs_compiler
def test_show_bad_source_errors(tmp_path: Path) -> None:
    src = tmp_path / "broken.c"
    src.write_text("int main(void) { return }\n")  # missing value

    result = runner.invoke(app, ["show", str(src)])
    assert result.exit_code == 1


def test_show_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.c"

    result = runner.invoke(app, ["show", str(missing)])
    assert result.exit_code == 1


def test_show_directory_is_rejected(tmp_path: Path) -> None:
    # passing a directory instead of a file should fail too
    result = runner.invoke(app, ["show", str(tmp_path)])
    assert result.exit_code == 1
