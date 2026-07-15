"""Tests for the diff command and its line-diffing engine.

The command itself is still a skeleton, so those tests just check it
validates its input and reports the levels it will compare. The
`diff_lines` tests below exercise the real diffing logic that the
rendering step will sit on top of.
"""

from pathlib import Path

from typer.testing import CliRunner

from compopt.cli import app
from compopt.diff import diff_lines, highlight_diff, render_diff

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


def test_diff_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.c"

    result = runner.invoke(app, ["diff", str(missing)])
    assert result.exit_code == 1


def test_diff_directory_is_rejected(tmp_path: Path) -> None:
    # a directory isn't a source file, so this should fail like a missing one
    result = runner.invoke(app, ["diff", str(tmp_path)])
    assert result.exit_code == 1
