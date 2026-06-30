"""Tests for the show command."""

from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from compopt.cli import app
from compopt.compilers import find_compilers
from compopt.show import (
    ALL_LEVELS,
    NARROW_LEVELS,
    highlight_asm,
    levels_for_width,
    line_number_gutter,
    render_columns,
)

runner = CliRunner()


def _render_to_text(table) -> str:
    # render the table the way the terminal would, then read the text back
    console = Console(width=120, record=True)
    console.print(table)
    return console.export_text()

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


def test_render_columns_keeps_headers_and_bodies() -> None:
    table = render_columns([("-O0", "add:\n\tret"), ("-O2", "add:\n\tret")])
    # column 0 is the line-number gutter, so the levels start at column 1
    assert table.columns[1].header == "-O0"
    assert table.columns[2].header == "-O2"

    text = _render_to_text(table)
    assert "-O0" in text
    assert "-O2" in text
    # the actual asm has to make it into the cells, not just the headers
    assert "add:" in text
    assert "ret" in text


def test_render_columns_shows_both_levels_distinctly() -> None:
    table = render_columns([("-O0", "pushq\t%rbp"), ("-O2", "ret")])
    text = _render_to_text(table)
    # each column carries its own body, side by side
    assert "pushq" in text
    assert "ret" in text


def test_line_number_gutter_numbers_every_row() -> None:
    gutter = line_number_gutter(3)
    assert gutter.plain.split("\n") == ["1", "2", "3"]


def test_line_number_gutter_right_aligns_numbers() -> None:
    # once the count hits double digits the single digits get padded so the
    # ones line up under the tens
    lines = line_number_gutter(10).plain.split("\n")
    assert lines[0] == " 1"
    assert lines[9] == "10"


def test_line_number_gutter_empty_body() -> None:
    assert line_number_gutter(0).plain == ""


def test_render_columns_truncates_long_lines() -> None:
    # a line that overruns its column should get chopped with an ellipsis
    # instead of wrapping onto a second row
    long_line = "movq " + "z" * 200
    table = render_columns([("-O0", long_line), ("-O2", "ret")])
    console = Console(width=40, record=True)
    console.print(table)
    text = console.export_text()
    assert "…" in text
    # the tail of the long line shouldn't survive the cut
    assert "z" * 200 not in text


def test_render_columns_shows_line_numbers() -> None:
    table = render_columns([("-O0", "add:\n\tret"), ("-O2", "add:\n\tret")])
    text = _render_to_text(table)
    # the two-line bodies should pick up numbers 1 and 2 in the margin
    assert "1" in text
    assert "2" in text


def test_levels_for_width_wide_shows_all_four() -> None:
    # a big terminal has room for the full -O0..-O3 comparison
    assert levels_for_width(200) == ALL_LEVELS


def test_levels_for_width_narrow_falls_back() -> None:
    # too tight for four columns, so drop back to the two-level view
    assert levels_for_width(40) == NARROW_LEVELS


def test_levels_for_width_never_goes_below_two() -> None:
    # even a silly-small width still gives us something to compare
    assert len(levels_for_width(1)) >= 2


def _styles(text) -> set[str]:
    # the colors rich applied, as plain strings we can assert on
    return {str(span.style) for span in text.spans}


def test_highlight_asm_keeps_the_text() -> None:
    body = "add:\n\tmovq %rbp, %rax"
    # coloring must not change a single character of the assembly
    assert highlight_asm(body).plain == body


def test_highlight_asm_colors_instruction_tokens() -> None:
    text = highlight_asm("\tmovq $5, %rax")
    styles = _styles(text)
    # mnemonic, immediate and register each get their own color
    assert "green" in styles
    assert "yellow" in styles
    assert "magenta" in styles


def test_highlight_asm_marks_labels() -> None:
    text = highlight_asm("add:")
    assert any("cyan" in style for style in _styles(text))


def test_highlight_asm_leaves_comments_plain() -> None:
    # a comment line carries no instruction, so nothing should be colored
    assert not highlight_asm("# this is a comment").spans


def test_highlight_asm_no_color_drops_styling() -> None:
    body = "\tmovq $5, %rax"
    text = highlight_asm(body, color=False)
    # the text survives untouched but nothing is styled
    assert text.plain == body
    assert not text.spans


def test_line_number_gutter_no_color_drops_styling() -> None:
    gutter = line_number_gutter(3, color=False)
    assert gutter.plain.split("\n") == ["1", "2", "3"]
    # without color the dim style shouldn't be attached to any number
    assert not gutter.spans


def test_render_columns_no_color_has_no_styled_cells() -> None:
    table = render_columns([("-O0", "add:\n\tret")], color=False)
    # every cell should come through as plain text with no spans
    gutter = next(iter(table.columns[0].cells))
    body = next(iter(table.columns[1].cells))
    assert not gutter.spans
    assert not body.spans


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
