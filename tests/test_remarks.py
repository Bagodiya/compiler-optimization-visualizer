"""Tests for getting clang's -Rpass remarks out of a compile, and reading them."""

import subprocess
from pathlib import Path

import pytest

from compopt import remarks
from compopt.compilers import CompileError, find_compilers
from compopt.remarks import (
    NO_CARET_FLAG,
    REMARK_FLAGS,
    RemarksUnsupported,
    capture_remarks,
    parse_remark,
    parse_remarks,
    rejected_the_flags,
    remark_blocks,
)
from compopt.report import MISSED, NOTE, OPTIMIZED

ADD_C = "int add(int a, int b) { return a + b; }\n"

# a loop clang vectorizes and a static function it inlines, so there's
# something for all three sorts of remark to be about
LOOP_C = """static int helper(int a, int b) { return a * b; }

int total(const int *xs, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += xs[i];
    return sum;
}

int caller(int n) { return helper(n, 3) + helper(n, 4); }
"""


def fake_run(stderr: str = "", returncode: int = 0):
    """Stand in for subprocess.run so the tests don't need a real clang.

    Keeps every command it was called with in `calls`, for the tests that
    care what we asked the compiler for.
    """
    calls: list[list[str]] = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)

    run.calls = calls
    return run


def real_clang() -> str | None:
    """A clang on this machine that really takes -Rpass, if there is one.

    On macOS `gcc` is Apple clang wearing gcc's name, so this usually finds
    one under whichever name comes first.
    """
    for name in find_compilers():
        probe = subprocess.run(
            [name, REMARK_FLAGS[0], "--version"], capture_output=True, text=True
        )
        if probe.returncode == 0:
            return name
    return None


# telling "you passed a flag I don't know" apart from "your C is broken"


def test_spots_gcc_turning_the_flags_down() -> None:
    msg = "gcc: error: unrecognized command-line option '-Rpass=.*'"
    assert rejected_the_flags(msg)


def test_a_broken_source_is_not_a_rejected_flag() -> None:
    msg = "bad.c:1:17: error: use of undeclared identifier 'this'"
    assert not rejected_the_flags(msg)


def test_unsupported_remembers_which_compiler() -> None:
    err = RemarksUnsupported("gcc")
    assert err.compiler == "gcc"
    assert "-Rpass" in str(err)


# the capture itself, against a fake compiler


def test_reads_the_remarks_off_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "loop.c"
    src.write_text(LOOP_C)
    said = "loop.c:5:5: remark: vectorized loop [-Rpass=loop-vectorize]\n"
    monkeypatch.setattr(remarks.subprocess, "run", fake_run(stderr=said))

    assert capture_remarks(src, "2", "clang") == said


def test_builds_the_command_we_meant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "loop.c"
    src.write_text(LOOP_C)
    run = fake_run()
    monkeypatch.setattr(remarks.subprocess, "run", run)

    capture_remarks(src, "3", "clang")

    cmd = run.calls[0]
    assert cmd[0] == "clang"
    assert "-S" in cmd
    assert "-O3" in cmd
    assert str(src) in cmd
    # all three sorts, or half the passes stay quiet
    for flag in REMARK_FLAGS:
        assert flag in cmd
    assert NO_CARET_FLAG in cmd


def test_rejected_flags_become_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "add.c"
    src.write_text(ADD_C)
    monkeypatch.setattr(
        remarks.subprocess,
        "run",
        fake_run(returncode=1, stderr="gcc: error: unrecognized command-line option '-Rpass=.*'"),
    )

    with pytest.raises(RemarksUnsupported) as excinfo:
        capture_remarks(src, "2", "gcc")

    assert excinfo.value.compiler == "gcc"


def test_broken_source_becomes_compile_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "broken.c"
    src.write_text("int main(void) { this is not c }\n")
    monkeypatch.setattr(
        remarks.subprocess,
        "run",
        fake_run(returncode=1, stderr="broken.c:1:18: error: expected ';'"),
    )

    with pytest.raises(CompileError) as excinfo:
        capture_remarks(src, "2", "clang")

    assert excinfo.value.compiler == "clang"
    assert "expected ';'" in excinfo.value.message


def test_failure_with_no_output_still_says_something(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "add.c"
    src.write_text(ADD_C)
    monkeypatch.setattr(remarks.subprocess, "run", fake_run(returncode=1))

    with pytest.raises(CompileError) as excinfo:
        capture_remarks(src, "2", "clang")

    assert excinfo.value.message


def test_temp_dir_does_not_stay_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(remarks.tempfile, "tempdir", str(scratch))

    src = tmp_path / "loop.c"
    src.write_text(LOOP_C)
    monkeypatch.setattr(remarks.subprocess, "run", fake_run(stderr="something\n"))

    capture_remarks(src, "2", "clang")

    assert list(scratch.iterdir()) == []


# reading the remarks back


# out of `clang -O3 -Rpass=.* -Rpass-missed=.* -Rpass-analysis=.*
# -fno-caret-diagnostics` on LOOP_C. The wording of the longer messages is cut
# down to fit, but nothing is tidied — the wrapped asm-printer remark and the
# warning sitting in the middle of it all are both really in there.
REAL_REMARKS = """loop.c:10:28: remark: 'helper' inlined into 'caller' (cost=-35) [-Rpass=inline]
loop.c:5:5: remark: vectorized loop (vectorization width: 4) [-Rpass=loop-vectorize]
loop.c:3:1: remark: 3 virtual registers copies generated in function [-Rpass-missed=regalloc]
loop.c:3:5: remark: Loop Strength Reduction: count changed 95 to 96 [-Rpass-analysis=size-info]
loop.c:1:18: warning: unused variable 'x' [-Wunused-variable]
loop.c:5:5: remark: BasicBlock:
cmpl\t: 1
movl\t: 1
 [-Rpass-analysis=asm-printer]
"""


def test_reads_a_pass_that_fired() -> None:
    line = "loop.c:5:5: remark: vectorized loop (width: 4) [-Rpass=loop-vectorize]"
    record = parse_remark(line)

    assert record is not None
    assert record.kind == OPTIMIZED
    assert record.file == "loop.c"
    assert record.line == 5
    assert record.column == 5
    assert record.message == "vectorized loop (width: 4)"
    assert record.pass_name == "loop-vectorize"
    assert record.helped


def test_reads_a_missed_pass() -> None:
    line = "loop.c:3:1: remark: 3 virtual registers copies [-Rpass-missed=regalloc]"
    record = parse_remark(line)

    assert record is not None
    assert record.kind == MISSED
    assert record.pass_name == "regalloc"
    assert not record.helped


def test_an_analysis_remark_is_a_note() -> None:
    line = "loop.c:3:5: remark: Loop Strength Reduction: Delta: 1 [-Rpass-analysis=size-info]"
    record = parse_remark(line)

    assert record is not None
    assert record.kind == NOTE
    assert record.pass_name == "size-info"


def test_column_is_optional() -> None:
    record = parse_remark("loop.c:5: remark: unrolled loop [-Rpass=loop-unroll]")

    assert record is not None
    assert record.line == 5
    assert record.column is None


def test_a_path_with_a_colon_in_it_still_parses() -> None:
    record = parse_remark("/tmp/odd:name/loop.c:5:5: remark: vectorized [-Rpass=loop-vectorize]")

    assert record is not None
    assert record.file == "/tmp/odd:name/loop.c"
    assert record.line == 5


def test_a_remark_with_no_tag_is_dropped() -> None:
    # nothing says whether this one fired or gave up, and guessing from the
    # wording is the reading-backwards this half of the tool is here to avoid
    assert parse_remark("loop.c:5:5: remark: vectorized loop") is None


def test_a_warning_is_not_a_remark() -> None:
    assert parse_remark("loop.c:1:18: warning: unused variable 'x' [-Wunused-variable]") is None


def test_a_wrapped_remark_keeps_its_whole_message() -> None:
    block = "loop.c:5:5: remark: BasicBlock: \ncmpl\t: 1\n [-Rpass-analysis=asm-printer]"
    record = parse_remark(block)

    assert record is not None
    assert record.pass_name == "asm-printer"
    assert "BasicBlock:" in record.message
    assert "cmpl" in record.message


def test_blocks_keep_a_wrapped_remark_together() -> None:
    blocks = remark_blocks(REAL_REMARKS)

    # four one-liners plus the wrapped asm-printer one; the warning is neither
    assert len(blocks) == 5
    assert blocks[-1].startswith("loop.c:5:5: remark: BasicBlock:")
    assert blocks[-1].endswith("[-Rpass-analysis=asm-printer]")


def test_a_warning_ends_the_block_before_it() -> None:
    text = (
        "loop.c:5:5: remark: BasicBlock: \n"
        "cmpl\t: 1\n"
        "loop.c:1:18: warning: unused variable 'x' [-Wunused-variable]\n"
    )

    blocks = remark_blocks(text)

    assert len(blocks) == 1
    assert "warning" not in blocks[0]


def test_parses_a_whole_run() -> None:
    records = parse_remarks(REAL_REMARKS)

    assert [r.kind for r in records] == [OPTIMIZED, OPTIMIZED, MISSED, NOTE, NOTE]
    assert [r.pass_name for r in records] == [
        "inline",
        "loop-vectorize",
        "regalloc",
        "size-info",
        "asm-printer",
    ]


def test_remark_order_is_kept() -> None:
    records = parse_remarks(REAL_REMARKS)

    # inlining runs long before the vectorizer, and so it should still read
    assert "inlined" in records[0].message
    assert "vectorized" in records[1].message


def test_repeated_remarks_are_all_kept() -> None:
    # size-info reports a count for the module and then the same one for the
    # function. both are true, so neither gets folded away.
    repeats = "\n".join(
        f"loop.c:3:5: remark: count changed from {n} to {n + 1} [-Rpass-analysis=size-info]"
        for n in (95, 96, 99)
    )

    assert len(parse_remarks(repeats)) == 3


def test_an_empty_run_gives_nothing() -> None:
    assert parse_remarks("") == []


# and against whatever is really installed


def test_real_clang_remarks_on_a_loop(tmp_path: Path) -> None:
    clang = real_clang()
    if clang is None:
        pytest.skip("no clang that supports -Rpass on this machine")

    src = tmp_path / "loop.c"
    src.write_text(LOOP_C)

    records = parse_remarks(capture_remarks(src, "3", clang))

    assert records
    assert all(r.file.endswith("loop.c") for r in records)
    assert all(r.pass_name for r in records)
    # -O3 on a summable loop and a static function gives it plenty to say
    assert any(r.helped for r in records)


def test_real_clang_rejects_broken_source(tmp_path: Path) -> None:
    clang = real_clang()
    if clang is None:
        pytest.skip("no clang that supports -Rpass on this machine")

    src = tmp_path / "broken.c"
    src.write_text("int main(void) { this is not c }\n")

    with pytest.raises(CompileError):
        capture_remarks(src, "2", clang)


def test_real_clang_fires_nothing_at_O0(tmp_path: Path) -> None:
    clang = real_clang()
    if clang is None:
        pytest.skip("no clang that supports -Rpass on this machine")

    src = tmp_path / "add.c"
    src.write_text(ADD_C)

    records = parse_remarks(capture_remarks(src, "0", clang))

    # not empty, unlike gcc's report at -O0: the backend still has to select
    # instructions and lay a frame out, and those passes count their work out
    # loud. That none of them is an optimization firing is the real check.
    assert records
    assert not any(r.helped for r in records)
