"""Tests for getting gcc's -fopt-info-all report out of a compile."""

import subprocess
from pathlib import Path

import pytest

from compopt import report
from compopt.compilers import CompileError, find_compilers
from compopt.report import (
    OPT_INFO_FLAG,
    OptInfoUnsupported,
    capture_opt_info,
    rejected_the_flag,
)

ADD_C = "int add(int a, int b) { return a + b; }\n"

# a loop gcc has something to say about, so the report isn't empty
LOOP_C = """
int total(const int *xs, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += xs[i];
    return sum;
}
"""


def info_path(cmd: list[str]) -> Path:
    """Dig the file we asked the report to be written to back out of the command."""
    for arg in cmd:
        if arg.startswith(f"{OPT_INFO_FLAG}="):
            return Path(arg.split("=", 1)[1])
    raise AssertionError(f"no {OPT_INFO_FLAG}= in {cmd}")


def fake_run(writes: str | None = None, returncode: int = 0, stderr: str = ""):
    """Build a stand-in for subprocess.run so tests don't need a real gcc.

    Whatever it gets called with is kept in `calls` for the tests that care
    about the command line. `writes` is the text to leave in the report file,
    or None to write no file at all.
    """
    calls: list[list[str]] = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if writes is not None:
            info_path(cmd).write_text(writes)
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)

    run.calls = calls
    return run


def real_gcc() -> str | None:
    """The name of a gcc that really takes -fopt-info-all, if there is one.

    On macOS `gcc` is usually Apple clang wearing gcc's name, so this is None
    far more often than not and the tests below skip themselves.
    """
    for name in find_compilers():
        probe = subprocess.run(
            [name, f"{OPT_INFO_FLAG}=/dev/null", "--version"],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return name
    return None


# telling "you passed a bad flag" apart from "your C is broken"


def test_spots_the_clang_wording() -> None:
    msg = f"clang: error: unknown argument: '{OPT_INFO_FLAG}=/tmp/x.txt'"
    assert rejected_the_flag(msg)


def test_spots_the_gcc_wording() -> None:
    msg = f"gcc: error: unrecognized command-line option '{OPT_INFO_FLAG}'"
    assert rejected_the_flag(msg)


def test_a_broken_source_is_not_a_bad_flag() -> None:
    # this one is the source's fault and has to reach the caller as CompileError
    msg = "add.c:1:5: error: expected ';' after expression"
    assert not rejected_the_flag(msg)


def test_unsupported_remembers_which_compiler() -> None:
    err = OptInfoUnsupported("clang")
    assert err.compiler == "clang"
    assert OPT_INFO_FLAG in str(err)


# the capture itself, against a fake compiler


def test_reads_the_report_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "loop.c"
    src.write_text(LOOP_C)
    monkeypatch.setattr(report.subprocess, "run", fake_run(writes="loop.c:4:5: optimized: x\n"))

    assert capture_opt_info(src, "2", "gcc") == "loop.c:4:5: optimized: x\n"


def test_builds_the_command_we_meant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "loop.c"
    src.write_text(LOOP_C)
    run = fake_run(writes="")
    monkeypatch.setattr(report.subprocess, "run", run)

    capture_opt_info(src, "3", "gcc")

    cmd = run.calls[0]
    assert cmd[0] == "gcc"
    assert "-S" in cmd
    assert "-O3" in cmd
    assert str(src) in cmd
    # the report has to go to a file of ours, not to stderr with the warnings
    assert any(arg.startswith(f"{OPT_INFO_FLAG}=") for arg in cmd)


def test_nothing_written_is_an_empty_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "add.c"
    src.write_text(ADD_C)
    # writes=None: compiler said it was happy but left no file behind
    monkeypatch.setattr(report.subprocess, "run", fake_run(writes=None))

    assert capture_opt_info(src, "0", "gcc") == ""


def test_rejected_flag_becomes_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "add.c"
    src.write_text(ADD_C)
    monkeypatch.setattr(
        report.subprocess,
        "run",
        fake_run(returncode=1, stderr=f"clang: error: unknown argument: '{OPT_INFO_FLAG}=/x'"),
    )

    with pytest.raises(OptInfoUnsupported) as excinfo:
        capture_opt_info(src, "2", "clang")

    assert excinfo.value.compiler == "clang"


def test_broken_source_becomes_compile_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "broken.c"
    src.write_text("int main(void) { this is not c }\n")
    monkeypatch.setattr(
        report.subprocess,
        "run",
        fake_run(returncode=1, stderr="broken.c:1:18: error: expected ';'"),
    )

    with pytest.raises(CompileError) as excinfo:
        capture_opt_info(src, "2", "gcc")

    assert excinfo.value.compiler == "gcc"
    assert "expected ';'" in excinfo.value.message


def test_failure_with_no_output_still_says_something(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "add.c"
    src.write_text(ADD_C)
    monkeypatch.setattr(report.subprocess, "run", fake_run(returncode=1))

    with pytest.raises(CompileError) as excinfo:
        capture_opt_info(src, "2", "gcc")

    # an empty message would leave the user staring at a blank error
    assert excinfo.value.message


def test_temp_dir_does_not_stay_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(report.tempfile, "tempdir", str(scratch))

    src = tmp_path / "loop.c"
    src.write_text(LOOP_C)
    monkeypatch.setattr(report.subprocess, "run", fake_run(writes="something\n"))

    capture_opt_info(src, "2", "gcc")

    assert list(scratch.iterdir()) == []


# and against whatever is really installed


def test_clang_is_reported_as_unsupported(tmp_path: Path) -> None:
    if "clang" not in find_compilers():
        pytest.skip("no clang on this machine to test against")

    src = tmp_path / "loop.c"
    src.write_text(LOOP_C)

    # clang has -Rpass instead and doesn't know this flag at all
    with pytest.raises(OptInfoUnsupported):
        capture_opt_info(src, "2", "clang")


def test_real_gcc_reports_on_a_loop(tmp_path: Path) -> None:
    gcc = real_gcc()
    if gcc is None:
        pytest.skip("no gcc that supports -fopt-info-all on this machine")

    src = tmp_path / "loop.c"
    src.write_text(LOOP_C)

    text = capture_opt_info(src, "3", gcc)

    # -O3 on a summable loop gives gcc plenty of passes to talk about
    assert text.strip()
    assert "loop.c" in text


def test_real_gcc_rejects_broken_source(tmp_path: Path) -> None:
    gcc = real_gcc()
    if gcc is None:
        pytest.skip("no gcc that supports -fopt-info-all on this machine")

    src = tmp_path / "broken.c"
    src.write_text("int main(void) { this is not c }\n")

    with pytest.raises(CompileError):
        capture_opt_info(src, "2", gcc)
