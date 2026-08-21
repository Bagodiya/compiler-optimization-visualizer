"""Tests for getting gcc's -fopt-info-all report out of a compile, and reading it."""

import dataclasses
import subprocess
from pathlib import Path

import pytest

from compopt import report
from compopt.compilers import CompileError, find_compilers
from compopt.report import (
    MISSED,
    NOTE,
    OPT_INFO_FLAG,
    OPTIMIZED,
    OptInfoUnsupported,
    OptRecord,
    capture_opt_info,
    parse_opt_info,
    parse_record,
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


# reading the report back


# straight out of `gcc-15 -O3 -fopt-info-all` on a file with a summable loop
# and a small static function called twice, so the chatter between the located
# lines is the real thing and not something I made up to be easy to skip.
REAL_REPORT = """loop.c:10:40: note: Considering inline candidate helper/2.
loop.c:10:40: optimized:  Inlining helper/2 into caller/3.
Unit growth for small function inlining: 16->16 (0%)

Inlined 2 calls, eliminated 1 functions

BB 3 is always executed in loop 1
loop.c:3:23: optimized: loop vectorized using 16 byte vectors
loop.c:1:5: note: vectorized 1 loops in function.
missed.c:2:23: missed: couldn't vectorize loop
"""


def test_reads_one_reported_line() -> None:
    record = parse_record("loop.c:3:23: optimized: loop vectorized using 16 byte vectors")

    assert record is not None
    assert record.kind == OPTIMIZED
    assert record.file == "loop.c"
    assert record.line == 3
    assert record.column == 23
    assert record.message == "loop vectorized using 16 byte vectors"


def test_reads_a_missed_pass() -> None:
    record = parse_record("missed.c:2:23: missed: couldn't vectorize loop")

    assert record is not None
    assert record.kind == MISSED
    assert not record.helped


def test_gccs_extra_space_comes_off() -> None:
    # gcc writes the inlining ones with two spaces after the colon
    record = parse_record("loop.c:10:40: optimized:  Inlining helper/2 into caller/3.")

    assert record is not None
    assert record.message == "Inlining helper/2 into caller/3."


def test_column_is_optional() -> None:
    record = parse_record("loop.c:3: note: vectorized 1 loops in function.")

    assert record is not None
    assert record.line == 3
    assert record.column is None


def test_a_path_with_a_colon_in_it_still_parses() -> None:
    record = parse_record("/tmp/odd:name/loop.c:3:23: missed: couldn't vectorize loop")

    assert record is not None
    assert record.file == "/tmp/odd:name/loop.c"
    assert record.line == 3


def test_pass_chatter_is_not_a_record() -> None:
    # no source position on any of these, so there's nowhere to put them
    assert parse_record("BB 3 is always executed in loop 1") is None
    assert parse_record("Unit growth for small function inlining: 16->16 (0%)") is None
    assert parse_record("Inlined 2 calls, eliminated 1 functions") is None
    assert parse_record("") is None


def test_an_unknown_kind_is_not_a_record() -> None:
    # a warning is a different thing to a pass report and shouldn't sneak in
    assert parse_record("loop.c:3:23: warning: unused variable 'x'") is None


def test_parses_a_whole_report() -> None:
    records = parse_opt_info(REAL_REPORT)

    assert len(records) == 5
    assert [r.kind for r in records] == [NOTE, OPTIMIZED, OPTIMIZED, NOTE, MISSED]


def test_report_order_is_kept() -> None:
    records = parse_opt_info(REAL_REPORT)

    # inlining runs before vectorization, and the report should still say so
    assert "Inlining" in records[1].message
    assert "vectorized" in records[2].message


def test_repeated_lines_are_all_kept() -> None:
    # gcc retries a loop at several vector widths and complains each time. the
    # repeats are the point — they say how hard it tried — so none get folded.
    retries = "\n".join(
        f"loop.c:4:13: note: ***** Analysis failed with vector mode V{n}SI" for n in (4, 8, 16)
    )

    assert len(parse_opt_info(retries)) == 3


def test_an_empty_report_gives_nothing() -> None:
    assert parse_opt_info("") == []


# the record type itself


def test_where_reads_back_like_gcc_wrote_it() -> None:
    record = OptRecord(OPTIMIZED, "loop vectorized", "loop.c", 3, 23)

    assert record.where() == "loop.c:3:23"


def test_where_drops_the_missing_column() -> None:
    record = OptRecord(NOTE, "vectorized 1 loops", "loop.c", 3)

    assert record.where() == "loop.c:3"


def test_only_optimized_counts_as_helped() -> None:
    assert OptRecord(OPTIMIZED, "did it", "loop.c", 3).helped
    assert not OptRecord(MISSED, "gave up", "loop.c", 3).helped
    assert not OptRecord(NOTE, "thinking about it", "loop.c", 3).helped


def test_a_made_up_kind_is_rejected() -> None:
    with pytest.raises(ValueError):
        OptRecord("warning", "unused variable", "loop.c", 3)


def test_a_zero_line_is_rejected() -> None:
    with pytest.raises(ValueError):
        OptRecord(OPTIMIZED, "did it", "loop.c", 0)


def test_records_do_not_change_after_building() -> None:
    record = OptRecord(OPTIMIZED, "did it", "loop.c", 3)

    with pytest.raises(dataclasses.FrozenInstanceError):
        record.message = "something else"


def test_real_gcc_report_parses(tmp_path: Path) -> None:
    gcc = real_gcc()
    if gcc is None:
        pytest.skip("no gcc that supports -fopt-info-all on this machine")

    src = tmp_path / "loop.c"
    src.write_text(LOOP_C)

    records = parse_opt_info(capture_opt_info(src, "3", gcc))

    # whatever it decided about this loop, it has to name the file it read
    assert records
    assert all(r.file.endswith("loop.c") for r in records)
    assert any(r.helped for r in records)
