"""Tests for the diff command and its line-diffing engine.

Most of this exercises the diffing and rendering pieces on hand-written asm,
where the expected output can be spelled out. The handful at the bottom drive
the command itself against a real compiler, so they only check the shape of
what comes back — the exact instructions are the compiler's business and
change between versions.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from compopt.cli import app
from compopt.compilers import find_compilers
from compopt.diff import (
    IDENTICAL_MESSAGE,
    NEITHER_MESSAGE,
    diff_lines,
    highlight_diff,
    is_identical,
    missing_message,
    render_diff,
    trim_context,
    unified_diff,
)

runner = CliRunner()

# the end-to-end ones need a real toolchain to produce anything to diff
needs_compiler = pytest.mark.skipif(
    not find_compilers(), reason="no gcc or clang available"
)

SOURCE = "int add(int a, int b) { return a + b; }\n"


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


def test_diff_lines_both_empty() -> None:
    # nothing on either side, so there's nothing to report
    assert diff_lines("", "") == []


def test_diff_lines_from_empty_is_all_additions() -> None:
    result = diff_lines("", "push rbp\nret")
    assert result == [("add", "push rbp"), ("add", "ret")]


def test_diff_lines_to_empty_is_all_removals() -> None:
    result = diff_lines("push rbp\nret", "")
    assert result == [("remove", "push rbp"), ("remove", "ret")]


def test_diff_lines_ignores_a_trailing_newline() -> None:
    # splitlines() already drops it, but the whole diff shifts by one if it
    # ever stops doing that, so pin it down here
    assert diff_lines("ret\n", "ret") == [("equal", "ret")]


def test_highlight_diff_colors_a_gap_line() -> None:
    text = highlight_diff([("gap", "4 unchanged lines")])
    assert text.plain == "@@ 4 unchanged lines"
    assert "cyan" in _styles(text)


def test_highlight_diff_gap_no_color_keeps_the_marker() -> None:
    text = highlight_diff([("gap", "4 unchanged lines")], color=False)
    assert text.plain == "@@ 4 unchanged lines"
    assert not text.spans


def test_is_identical_false_when_lines_were_folded_away() -> None:
    # a gap only shows up after trimming, and trimming only hides context
    # around a real change, so this is never "no difference"
    assert not is_identical([("gap", "4 unchanged lines"), ("add", "ret")])


def test_trim_context_all_equal_folds_into_one_gap() -> None:
    diff = [("equal", f"line {n}") for n in range(5)]
    assert trim_context(diff, context=3) == [("gap", "5 unchanged lines")]


def test_trim_context_with_no_equal_lines_adds_no_gap() -> None:
    diff = [("remove", "mov eax, 2"), ("add", "mov eax, 4")]
    assert trim_context(diff, context=2) == diff


def test_trim_context_then_render_reads_like_a_diff() -> None:
    lines = render_diff(trim_context(_sample_diff(), context=1)).splitlines()
    assert lines == [
        "@@ 5 unchanged lines",
        "  line 5",
        "+ line new",
        "  line 6",
        "@@ 5 unchanged lines",
    ]


def test_unified_diff_labels_default_to_the_usual_pair() -> None:
    text = unified_diff("mov eax, 2\nret", "mov eax, 4\nret")
    assert "--- O0" in text
    assert "+++ O2" in text


def test_unified_diff_from_empty_side() -> None:
    text = unified_diff("", "push rbp\nret")
    assert "+push rbp" in text
    assert "+ret" in text





def test_missing_message_none_when_both_sides_have_code() -> None:
    # the normal case, so nothing to explain and we go on to the real diff
    assert missing_message("push rbp\nret", "ret") is None


def test_missing_message_when_the_function_vanished() -> None:
    note = missing_message("push rbp\nret", "", from_level="0", to_level="2")
    assert note is not None
    assert "-O2" in note


def test_missing_message_when_it_only_appears_later() -> None:
    note = missing_message("", "push rbp\nret", from_level="0", to_level="3")
    assert note is not None
    assert "-O3" in note
    assert "-O0" in note


def test_missing_message_when_neither_level_has_it() -> None:
    assert missing_message("", "") == NEITHER_MESSAGE


def test_missing_message_treats_blank_lines_as_empty() -> None:
    # isolate_function can hand back a body that is only whitespace, and that
    # is still nothing worth diffing
    assert missing_message("\n  \n", "\t\n") == NEITHER_MESSAGE


def test_missing_message_beats_the_all_removed_diff() -> None:
    old = "push rbp\nmov rbp, rsp\npop rbp\nret"
    # without the note this would print four "-" lines and say nothing useful
    assert all(tag == "remove" for tag, _ in diff_lines(old, ""))
    assert missing_message(old, "") is not None


# from here down the command is driven for real, so these need a compiler

@needs_compiler
def test_diff_marks_lines_that_went_and_arrived(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text(SOURCE)

    result = runner.invoke(app, ["diff", str(src), "--no-color", "--width", "200"])
    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    # -O0 puts both arguments in memory and reads them back, -O2 doesn't, so
    # there is always something leaving and something arriving between the two
    assert any(line.startswith("-") for line in lines)
    assert any(line.startswith("+") for line in lines)


@needs_compiler
def test_diff_of_a_level_against_itself_says_so(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text(SOURCE)

    # pointless but not an error, and it should not print the body back
    result = runner.invoke(app, ["diff", str(src), "--from", "2", "--to", "2",
                                 "--no-color", "--width", "200"])
    assert result.exit_code == 0
    assert IDENTICAL_MESSAGE in result.stdout


@needs_compiler
def test_diff_accepts_the_o_spelling(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text(SOURCE)

    # --from O0 is how the level is spelled everywhere else, so it has to work
    spelled = runner.invoke(app, ["diff", str(src), "--from", "O0", "--to", "O2",
                                  "--no-color", "--width", "200"])
    bare = runner.invoke(app, ["diff", str(src), "--from", "0", "--to", "2",
                               "--no-color", "--width", "200"])
    assert spelled.exit_code == 0
    assert spelled.stdout == bare.stdout


@needs_compiler
def test_diff_unified_has_the_usual_header(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text(SOURCE)

    result = runner.invoke(app, ["diff", str(src), "-u", "--no-color", "--width", "200"])
    assert result.exit_code == 0
    assert "--- O0" in result.stdout
    assert "+++ O2" in result.stdout


@needs_compiler
def test_diff_context_controls_how_much_is_kept(tmp_path: Path) -> None:
    src = tmp_path / "loop.c"
    # something long enough that there are unchanged lines worth folding away
    src.write_text(
        "int sum(int n) { int t = 0; for (int i = 0; i < n; i++) t += i; return t; }\n"
    )

    tight = runner.invoke(app, ["diff", str(src), "-C", "0", "--no-color", "--width", "200"])
    loose = runner.invoke(app, ["diff", str(src), "-C", "9", "--no-color", "--width", "200"])
    assert tight.exit_code == 0
    assert loose.exit_code == 0
    # more context can only ever mean more lines on screen
    assert len(tight.stdout.splitlines()) <= len(loose.stdout.splitlines())


@needs_compiler
def test_diff_func_picks_the_function(tmp_path: Path) -> None:
    src = tmp_path / "two.c"
    src.write_text(
        "int add(int a, int b) { return a + b; }\n"
        "int sub(int a, int b) { return a - b; }\n"
    )

    result = runner.invoke(app, ["diff", str(src), "--func", "sub",
                                 "--no-color", "--width", "200"])
    assert result.exit_code == 0
    assert "sub" in result.stdout
    # asking for sub shouldn't drag add's body into the diff
    assert "add:" not in result.stdout


@needs_compiler
def test_diff_of_a_function_that_is_in_neither_level(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text(SOURCE)

    # a typo'd name is in neither build, which is the one message that covers it
    result = runner.invoke(app, ["diff", str(src), "--func", "nope", "--no-color"])
    assert result.exit_code == 0
    assert NEITHER_MESSAGE in result.stdout


@needs_compiler
def test_diff_bad_source_reports_the_compiler(tmp_path: Path) -> None:
    src = tmp_path / "broken.c"
    src.write_text("int main(void) { return }\n")

    result = runner.invoke(app, ["diff", str(src)])
    assert result.exit_code == 1
