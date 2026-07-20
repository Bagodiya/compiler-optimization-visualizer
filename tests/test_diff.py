"""Tests for the diff command and its line-diffing engine.

The command itself is still a skeleton, so those tests just check it
validates its input and reports the levels it will compare. The
`diff_lines` tests below exercise the real diffing logic that the
rendering step will sit on top of.
"""

from pathlib import Path

from typer.testing import CliRunner

from compopt.cli import app
from compopt.diff import (
    IDENTICAL_MESSAGE,
    diff_lines,
    highlight_diff,
    is_identical,
    render_diff,
    trim_context,
    unified_diff,
)

runner = CliRunner()


def _styles(text) -> set[str]:
    # the colors rich ended up attaching, as plain strings we can assert on
    return {str(span.style) for span in text.spans}


def test_diff_lines_identical_is_all_equal() -> None:
    asm = "push rbp\nmov rbp, rsp\npop rbp\nret"
    result = diff_lines(asm, asm)
    assert [tag for tag, _ in result] == ["equal"] * 4
    assert [line for _, line in result] == asm.splitlines()


def test_diff_lines_pure_addition() -> None:
    old = "mov eax, edi\nret"
    new = "mov eax, edi\nadd eax, 1\nret"
    result = diff_lines(old, new)
    assert ("add", "add eax, 1") in result
    # the shared lines stay equal, nothing is marked removed
    assert ("remove", "mov eax, edi") not in result
    assert result[0] == ("equal", "mov eax, edi")


def test_diff_lines_pure_removal() -> None:
    old = "push rbp\nmov rbp, rsp\npop rbp\nret"
    new = "ret"
    result = diff_lines(old, new)
    removed = [line for tag, line in result if tag == "remove"]
    assert "push rbp" in removed
    assert "mov rbp, rsp" in removed
    assert ("equal", "ret") in result


def test_diff_lines_replace_shows_remove_then_add() -> None:
    old = "mov eax, 2\nret"
    new = "mov eax, 4\nret"
    result = diff_lines(old, new)
    # a changed line reads as the old one leaving and the new one arriving,
    # and the removal must come before the addition
    tags = [tag for tag, _ in result]
    assert tags == ["remove", "add", "equal"]
    assert result[0] == ("remove", "mov eax, 2")
    assert result[1] == ("add", "mov eax, 4")


def test_render_diff_marks_each_line() -> None:
    diff = [
        ("equal", "mov eax, edi"),
        ("remove", "mov eax, 2"),
        ("add", "mov eax, 4"),
    ]
    lines = render_diff(diff).splitlines()
    assert lines == [
        "  mov eax, edi",
        "- mov eax, 2",
        "+ mov eax, 4",
    ]


def test_render_diff_end_to_end() -> None:
    # feed real diff_lines output straight into the renderer
    old = "mov eax, 2\nret"
    new = "mov eax, 4\nret"
    text = render_diff(diff_lines(old, new))
    assert "- mov eax, 2" in text
    assert "+ mov eax, 4" in text
    assert "  ret" in text


def test_render_diff_empty() -> None:
    assert render_diff([]) == ""


def test_highlight_diff_keeps_the_gutter_text() -> None:
    diff = [
        ("equal", "mov eax, edi"),
        ("remove", "mov eax, 2"),
        ("add", "mov eax, 4"),
    ]
    # coloring is only skin deep — the text should match the plain renderer
    assert highlight_diff(diff).plain == render_diff(diff)


def test_highlight_diff_colors_added_and_removed() -> None:
    diff = [
        ("equal", "ret"),
        ("remove", "mov eax, 2"),
        ("add", "mov eax, 4"),
    ]
    styles = _styles(highlight_diff(diff))
    # additions go green, removals go red
    assert "green" in styles
    assert "red" in styles


def test_highlight_diff_leaves_equal_lines_uncolored() -> None:
    # a line that didn't change is just context, so nothing should be tinted
    text = highlight_diff([("equal", "ret")])
    assert all(str(span.style) == "" for span in text.spans)


def test_highlight_diff_empty() -> None:
    text = highlight_diff([])
    assert text.plain == ""
    assert not text.spans


def test_highlight_diff_no_color_drops_styling() -> None:
    diff = [("remove", "mov eax, 2"), ("add", "mov eax, 4")]
    text = highlight_diff(diff, color=False)
    # the text survives but nothing is styled, same as --no-color elsewhere
    assert text.plain == render_diff(diff)
    assert not text.spans


def test_is_identical_when_nothing_changed() -> None:
    asm = "push rbp\nmov rbp, rsp\npop rbp\nret"
    assert is_identical(diff_lines(asm, asm))


def test_is_identical_false_when_a_line_moved() -> None:
    old = "mov eax, 2\nret"
    new = "mov eax, 4\nret"
    assert not is_identical(diff_lines(old, new))


def test_is_identical_false_for_an_empty_diff() -> None:
    # nothing to compare isn't the same as "compared and found no change"
    assert not is_identical([])


def test_render_diff_reports_identical_levels() -> None:
    asm = "push rbp\nmov rbp, rsp\nret"
    # one short line beats echoing the whole function back with a blank gutter
    assert render_diff(diff_lines(asm, asm)) == IDENTICAL_MESSAGE


def test_highlight_diff_reports_identical_levels() -> None:
    asm = "push rbp\nret"
    text = highlight_diff(diff_lines(asm, asm))
    assert text.plain == IDENTICAL_MESSAGE
    # no red or green here, nothing was added or removed
    assert "green" not in _styles(text)
    assert "red" not in _styles(text)


def test_highlight_diff_identical_no_color_still_says_so() -> None:
    asm = "push rbp\nret"
    text = highlight_diff(diff_lines(asm, asm), color=False)
    assert text.plain == IDENTICAL_MESSAGE
    assert not text.spans


def _sample_diff() -> list[tuple[str, str]]:
    # one change buried in a pile of unchanged lines, so trimming has
    # something real to fold away
    diff = [("equal", f"line {n}") for n in range(6)]
    diff.append(("add", "line new"))
    diff.extend(("equal", f"line {n}") for n in range(6, 12))
    return diff


def test_trim_context_keeps_lines_around_a_change() -> None:
    trimmed = trim_context(_sample_diff(), context=2)
    # the two equal lines on each side of the added line survive
    assert ("equal", "line 4") in trimmed
    assert ("equal", "line 5") in trimmed
    assert ("add", "line new") in trimmed
    assert ("equal", "line 6") in trimmed
    assert ("equal", "line 7") in trimmed
    # anything further out is gone
    assert ("equal", "line 3") not in trimmed
    assert ("equal", "line 8") not in trimmed


def test_trim_context_folds_hidden_lines_into_a_gap() -> None:
    trimmed = trim_context(_sample_diff(), context=2)
    gaps = [line for tag, line in trimmed if tag == "gap"]
    # four lines are hidden on each side (0-3 and 8-11)
    assert gaps == ["4 unchanged lines", "4 unchanged lines"]


def test_trim_context_zero_drops_all_equal_lines() -> None:
    trimmed = trim_context(_sample_diff(), context=0)
    assert ("add", "line new") in trimmed
    assert not any(tag == "equal" for tag, _ in trimmed)


def test_trim_context_negative_leaves_diff_untouched() -> None:
    diff = _sample_diff()
    assert trim_context(diff, context=-1) == diff


def test_trim_context_wide_enough_hides_nothing() -> None:
    diff = _sample_diff()
    trimmed = trim_context(diff, context=100)
    # nothing to fold, so no gaps and the diff comes back as-is
    assert trimmed == diff


def test_trim_context_gap_singular_wording() -> None:
    diff = [("add", "x"), ("equal", "solo"), ("add", "y")]
    trimmed = trim_context(diff, context=0)
    assert ("gap", "1 unchanged line") in trimmed


def test_render_diff_marks_a_gap_line() -> None:
    text = render_diff([("gap", "4 unchanged lines")])
    assert text == "@@ 4 unchanged lines"


def test_unified_diff_has_headers_and_a_hunk() -> None:
    old = "mov eax, 2\nret"
    new = "mov eax, 4\nret"
    text = unified_diff(old, new, from_label="O0", to_label="O2")
    # the standard unified header names both sides
    assert "--- O0" in text
    assert "+++ O2" in text
    # and a hunk marker with the changed lines under it
    assert "@@" in text
    assert "-mov eax, 2" in text
    assert "+mov eax, 4" in text
    # the unchanged line stays as plain context (leading space, no marker)
    assert " ret" in text


def test_unified_diff_identical_is_empty() -> None:
    # nothing changed, so there's no hunk to print and no header either
    asm = "push rbp\nmov rbp, rsp\nret"
    assert unified_diff(asm, asm) == ""


def test_unified_diff_respects_context() -> None:
    old = "\n".join(f"line {n}" for n in range(20))
    new = old + "\ntail"
    # one added line at the very end; with tight context the far-away lines
    # at the top shouldn't get pulled into the hunk
    text = unified_diff(old, new, context=1)
    assert "+tail" in text
    assert "line 19" in text
    assert "line 0" not in text


def test_diff_reports_levels(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    result = runner.invoke(app, ["diff", str(src)])
    assert result.exit_code == 0
    # the placeholder should name the two levels it's going to compare
    assert "-O0" in result.stdout
    assert "-O2" in result.stdout


def test_diff_uses_the_levels_it_was_given(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    result = runner.invoke(app, ["diff", str(src), "--from", "1", "--to", "3"])
    assert result.exit_code == 0
    assert "-O1" in result.stdout
    assert "-O3" in result.stdout
    # the defaults shouldn't leak through once the flags are set
    assert "-O0" not in result.stdout


def test_diff_rejects_a_level_we_cant_compile(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    result = runner.invoke(app, ["diff", str(src), "--from", "9"])
    assert result.exit_code == 1


def test_diff_rejects_a_bad_to_level(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    # -Ofast and friends aren't wired up yet, so this is still an error
    result = runner.invoke(app, ["diff", str(src), "--to", "fast"])
    assert result.exit_code == 1


def test_diff_reports_the_context_it_will_use(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    result = runner.invoke(app, ["diff", str(src), "--context", "5"])
    assert result.exit_code == 0
    assert "5 lines of context" in result.stdout


def test_diff_reports_unified_mode(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    result = runner.invoke(app, ["diff", str(src), "--unified"])
    assert result.exit_code == 0
    assert "unified" in result.stdout


def test_diff_rejects_negative_context(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    result = runner.invoke(app, ["diff", str(src), "--context", "-1"])
    assert result.exit_code == 1


def test_diff_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.c"

    result = runner.invoke(app, ["diff", str(missing)])
    assert result.exit_code == 1


def test_diff_directory_is_rejected(tmp_path: Path) -> None:
    # a directory isn't a source file, so this should fail like a missing one
    result = runner.invoke(app, ["diff", str(tmp_path)])
    assert result.exit_code == 1
