"""Tests for the show command.

The rendering helpers themselves live in test_render.py; here we drive the
command end to end and check the pieces are wired together.
"""

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from compopt.cli import app
from compopt.compilers import find_compilers
from compopt.show import _pick_compiler

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
def test_show_compiler_flag_picks_available(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    # ask for one we know is installed and make sure it goes through
    picked = find_compilers()[0]
    result = runner.invoke(app, ["show", str(src), "--compiler", picked])
    assert result.exit_code == 0
    assert "add" in result.stdout


@needs_compiler
def test_show_unknown_compiler_errors(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    # a compiler that isn't a real thing should fail before we compile anything
    result = runner.invoke(app, ["show", str(src), "--compiler", "notacc"])
    assert result.exit_code == 1
    assert "notacc" in result.stdout + result.stderr


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


# these poke _pick_compiler directly so they don't need a real toolchain

def test_pick_compiler_flag_beats_cc(monkeypatch: pytest.MonkeyPatch) -> None:
    # an explicit --compiler should ignore whatever $CC says
    monkeypatch.setenv("CC", "clang")
    assert _pick_compiler("gcc", ["gcc", "clang"]) == "gcc"


def test_pick_compiler_uses_cc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CC", "clang")
    assert _pick_compiler(None, ["gcc", "clang"]) == "clang"


def test_pick_compiler_cc_can_be_a_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # CC is often a full path, so we match on the file name
    monkeypatch.setenv("CC", "/usr/local/bin/clang")
    assert _pick_compiler(None, ["gcc", "clang"]) == "clang"


def test_pick_compiler_ignores_unusable_cc(monkeypatch: pytest.MonkeyPatch) -> None:
    # CC=cc isn't something we know how to drive, so fall back to gcc-first
    monkeypatch.setenv("CC", "cc")
    assert _pick_compiler(None, ["gcc", "clang"]) == "gcc"


def test_pick_compiler_default_when_no_cc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CC", raising=False)
    assert _pick_compiler(None, ["gcc", "clang"]) == "gcc"


def test_pick_compiler_bad_flag_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CC", raising=False)
    with pytest.raises(typer.Exit):
        _pick_compiler("notacc", ["gcc", "clang"])
