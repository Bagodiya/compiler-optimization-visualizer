"""Tests for the side-by-side rendering helpers.

These exercise the pure rendering functions in isolation — no compiler needed,
we just feed them assembly text and check what comes back out.
"""

from rich.console import Console

from compopt.render import (
    ALL_LEVELS,
    MIN_COLUMN_WIDTH,
    NARROW_LEVELS,
    highlight_asm,
    levels_for_width,
    line_number_gutter,
    render_columns,
)


def _render_to_text(table) -> str:
    # render the table the way the terminal would, then read the text back
    console = Console(width=120, record=True)
    console.print(table)
    return console.export_text()


def _styles(text) -> set[str]:
    # the colors rich applied, as plain strings we can assert on
    return {str(span.style) for span in text.spans}


# --- levels_for_width -------------------------------------------------------


def test_levels_for_width_wide_shows_all_four() -> None:
    # a big terminal has room for the full -O0..-O3 comparison
    assert levels_for_width(200) == ALL_LEVELS


def test_levels_for_width_narrow_falls_back() -> None:
    # too tight for four columns, so drop back to the two-level view
    assert levels_for_width(40) == NARROW_LEVELS


def test_levels_for_width_never_goes_below_two() -> None:
    # even a silly-small width still gives us something to compare
    assert len(levels_for_width(1)) >= 2


def test_levels_for_width_boundary_is_inclusive() -> None:
    # right on the threshold there's just enough room for all four; one column
    # short of it we fall back
    threshold = MIN_COLUMN_WIDTH * len(ALL_LEVELS)
    assert levels_for_width(threshold) == ALL_LEVELS
    assert levels_for_width(threshold - 1) == NARROW_LEVELS


# --- highlight_asm ----------------------------------------------------------


def test_highlight_asm_keeps_the_text() -> None:
    body = "add:\n\tmovq %rbp, %rax"
    # coloring must not change a single character of the assembly
    assert highlight_asm(body).plain == body


def test_highlight_asm_empty_body() -> None:
    text = highlight_asm("")
    assert text.plain == ""
    assert not text.spans


def test_highlight_asm_keeps_blank_lines() -> None:
    body = "add:\n\n\tret"
    # a blank line in the middle has to survive so the numbering stays aligned
    assert highlight_asm(body).plain == body


def test_highlight_asm_colors_instruction_tokens() -> None:
    text = highlight_asm("\tmovq $5, %rax")
    styles = _styles(text)
    # mnemonic, immediate and register each get their own color
    assert "green" in styles
    assert "yellow" in styles
    assert "magenta" in styles


def test_highlight_asm_colors_negative_immediate() -> None:
    # negative offsets like $-8 are still immediates and should be picked up
    text = highlight_asm("\tsubq $-8, %rsp")
    assert "yellow" in _styles(text)


def test_highlight_asm_marks_labels() -> None:
    text = highlight_asm("add:")
    assert any("cyan" in style for style in _styles(text))


def test_highlight_asm_marks_local_labels() -> None:
    # the compiler's own .L labels are still labels as far as color goes
    text = highlight_asm(".L2:")
    assert any("cyan" in style for style in _styles(text))


def test_highlight_asm_label_stops_before_trailing_comment() -> None:
    # clang writes "add:  ## @add"; only the label part should get colored,
    # the comment is left alone
    line = "add:  ## @add"
    text = highlight_asm(line)
    cyan = [s for s in text.spans if "cyan" in str(s.style)]
    assert cyan
    # the colored span ends at the colon, well before the "##" comment
    assert all(s.end <= line.index("#") for s in cyan)


def test_highlight_asm_leaves_comments_plain() -> None:
    # a comment line carries no instruction, so nothing should be colored
    assert not highlight_asm("# this is a comment").spans


def test_highlight_asm_no_color_drops_styling() -> None:
    body = "\tmovq $5, %rax"
    text = highlight_asm(body, color=False)
    # the text survives untouched but nothing is styled
    assert text.plain == body
    assert not text.spans


# --- line_number_gutter -----------------------------------------------------


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


def test_line_number_gutter_no_color_drops_styling() -> None:
    gutter = line_number_gutter(3, color=False)
    assert gutter.plain.split("\n") == ["1", "2", "3"]
    # without color the dim style shouldn't be attached to any number
    assert not gutter.spans


# --- render_columns ---------------------------------------------------------


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


def test_render_columns_shows_line_numbers() -> None:
    table = render_columns([("-O0", "add:\n\tret"), ("-O2", "add:\n\tret")])
    text = _render_to_text(table)
    # the two-line bodies should pick up numbers 1 and 2 in the margin
    assert "1" in text
    assert "2" in text


def test_render_columns_numbers_up_to_the_longest_body() -> None:
    # one column has three lines, the other has one; the gutter should count
    # the taller of the two so every row of the long column is numbered
    table = render_columns([("-O0", "a\nb\nc"), ("-O2", "x")])
    gutter = next(iter(table.columns[0].cells))
    assert gutter.plain.split("\n") == ["1", "2", "3"]


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


def test_render_columns_empty_list_is_harmless() -> None:
    # nothing to show shouldn't blow up — we just get the gutter column and
    # no rows to speak of
    table = render_columns([])
    assert len(table.columns) == 1
    gutter = next(iter(table.columns[0].cells))
    assert gutter.plain == ""


def test_render_columns_no_color_has_no_styled_cells() -> None:
    table = render_columns([("-O0", "add:\n\tret")], color=False)
    # every cell should come through as plain text with no spans
    gutter = next(iter(table.columns[0].cells))
    body = next(iter(table.columns[1].cells))
    assert not gutter.spans
    assert not body.spans
