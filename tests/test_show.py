"""Tests for the show command.

The rendering helpers themselves live in test_render.py; here we drive the
command end to end and check the pieces are wired together.
"""

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


@needs_compiler
def test_show_renders_both_levels(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    result = runner.invoke(app, ["show", str(src)])
    assert result.exit_code == 0
    # both optimization levels should be labelled in the side-by-side view
    assert "-O0" in result.stdout
    assert "-O2" in result.stdout


@needs_compiler
def test_show_no_color_runs_clean(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    result = runner.invoke(app, ["show", str(src), "--no-color"])
    assert result.exit_code == 0
    # the asm still shows up, we've only dropped the highlighting
    assert "add" in result.stdout


@needs_compiler
def test_show_wide_width_forces_all_levels(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    # a big --width should give us the full four-column view no matter how
    # wide the test terminal actually is
    result = runner.invoke(app, ["show", str(src), "--width", "200"])
    assert result.exit_code == 0
    assert "-O1" in result.stdout
    assert "-O3" in result.stdout


@needs_compiler
def test_show_narrow_width_forces_two_levels(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    # too tight for four columns, so we should fall back to -O0 vs -O2
    result = runner.invoke(app, ["show", str(src), "--width", "40"])
    assert result.exit_code == 0
    assert "-O0" in result.stdout
    assert "-O2" in result.stdout
    assert "-O1" not in result.stdout
    assert "-O3" not in result.stdout
