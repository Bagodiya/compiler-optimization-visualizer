"""Tests for the diff command and its line-diffing engine.

The command itself is still a skeleton, so those tests just check it
validates its input and reports the levels it will compare. The
`diff_lines` tests below exercise the real diffing logic that the
rendering step will sit on top of.
"""

from pathlib import Path

from typer.testing import CliRunner

from compopt.cli import app
from compopt.diff import diff_lines

runner = CliRunner()


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
