"""Tests for picking a way to ask the compiler what its passes did.

Most of these stand in for the two capture functions, since what's being
tested is the choosing rather than either capture — those have their own
tests in test_report.py and test_remarks.py. The ones at the bottom run the
whole thing against whatever compiler is really installed, which is the only
place the fallback gets exercised end to end.
"""

from pathlib import Path

import pytest

from compopt import passes
from compopt.compilers import CompileError, find_compilers
from compopt.passes import (
    FLAGS,
    OPT_INFO,
    REMARKS,
    RPASS_FLAG,
    NoPassReport,
    PassReport,
    collect_report,
    kind_label,
)
from compopt.remarks import RemarksUnsupported
from compopt.report import (
    MISSED,
    NOTE,
    OPT_INFO_FLAG,
    OPTIMIZED,
    OptInfoUnsupported,
    OptRecord,
)

needs_compiler = pytest.mark.skipif(
    not find_compilers(), reason="no gcc or clang available"
)

SUM_C = """int total(const int *xs, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += xs[i];
    return sum;
}
"""

GCC_TEXT = """sum.c:3:5: optimized: loop vectorized using 16 byte vectors
sum.c:5:12: missed: statement clobbers memory
"""

CLANG_TEXT = """sum.c:3:5: remark: vectorized loop (width: 4) [-Rpass=loop-vectorize]
sum.c:5:12: remark: 3 spills generated [-Rpass-missed=regalloc]
"""


def answers(text: str):
    """A capture function that always hands back the same text."""
    return lambda source, level, compiler: text


def refuses(error):
    """A capture function that always turns the flag down."""

    def capture(source, level, compiler):
        raise error(compiler)

    return capture


def use(monkeypatch, opt_info, remarks) -> None:
    """Put both capture functions in place at once.

    Always both, because the whole point of `collect_report` is that it may
    call either one, and a test that only patches the half it expects will
    quietly shell out to a real compiler when it's wrong.
    """
    monkeypatch.setattr(passes, "capture_opt_info", opt_info)
    monkeypatch.setattr(passes, "capture_remarks", remarks)


@pytest.fixture
def sum_c(tmp_path: Path) -> Path:
    src = tmp_path / "sum.c"
    src.write_text(SUM_C)
    return src


# which way it ends up asking


def test_gcc_is_asked_through_opt_info(sum_c: Path, monkeypatch) -> None:
    use(monkeypatch, answers(GCC_TEXT), refuses(RemarksUnsupported))

    report = collect_report(sum_c, "2", "gcc")

    assert report.via == OPT_INFO
    assert report.flag == OPT_INFO_FLAG
    assert [r.kind for r in report.records] == [OPTIMIZED, MISSED]


def test_a_refused_flag_falls_through_to_the_other(sum_c: Path, monkeypatch) -> None:
    # the everyday macOS case: the compiler is named gcc and is really clang
    use(monkeypatch, refuses(OptInfoUnsupported), answers(CLANG_TEXT))

    report = collect_report(sum_c, "2", "gcc")

    assert report.via == REMARKS
    assert report.flag == RPASS_FLAG
    assert [r.pass_name for r in report.records] == ["loop-vectorize", "regalloc"]


def test_neither_way_working_is_its_own_error(sum_c: Path, monkeypatch) -> None:
    use(monkeypatch, refuses(OptInfoUnsupported), refuses(RemarksUnsupported))

    with pytest.raises(NoPassReport) as excinfo:
        collect_report(sum_c, "2", "cc")

    assert excinfo.value.compiler == "cc"
    assert OPT_INFO_FLAG in str(excinfo.value)
    assert RPASS_FLAG in str(excinfo.value)


def test_the_second_way_is_left_alone_when_the_first_answers(
    sum_c: Path, monkeypatch
) -> None:
    def never(source, level, compiler):
        raise AssertionError("asked clang after gcc had already answered")

    use(monkeypatch, answers(GCC_TEXT), never)

    assert collect_report(sum_c, "2", "gcc").records


def test_broken_source_is_not_retried_the_other_way(sum_c: Path, monkeypatch) -> None:
    # source that doesn't build doesn't build either way, so compiling it a
    # second time to be told so again would just double the wait
    def broken(source, level, compiler):
        raise CompileError(compiler, "sum.c:1:5: error: expected ';'")

    def never(source, level, compiler):
        raise AssertionError("recompiled source that had already failed")

    use(monkeypatch, broken, never)

    with pytest.raises(CompileError):
        collect_report(sum_c, "2", "gcc")


def test_a_compiler_with_nothing_to_say_still_gives_a_report(
    sum_c: Path, monkeypatch
) -> None:
    # gcc at -O0 writes an empty report rather than refusing, and that's an
    # answer: no passes ran
    use(monkeypatch, answers(""), refuses(RemarksUnsupported))

    report = collect_report(sum_c, "0", "gcc")

    assert report.records == ()
    assert report.via == OPT_INFO


def test_the_level_asked_for_is_kept(sum_c: Path, monkeypatch) -> None:
    use(monkeypatch, answers(GCC_TEXT), refuses(RemarksUnsupported))

    assert collect_report(sum_c, "3", "gcc").level == "3"


# the report itself


def test_every_way_of_asking_has_a_flag_to_name_it() -> None:
    assert set(FLAGS) == {OPT_INFO, REMARKS}


def test_a_way_we_do_not_have_is_refused() -> None:
    with pytest.raises(ValueError):
        PassReport("gcc", "2", "guesswork")


def test_records_can_be_taken_a_kind_at_a_time() -> None:
    records = (
        OptRecord(kind=OPTIMIZED, message="vectorized", file="sum.c", line=3),
        OptRecord(kind=MISSED, message="gave up", file="sum.c", line=4),
        OptRecord(kind=NOTE, message="thinking", file="sum.c", line=5),
    )
    report = PassReport("clang", "2", REMARKS, records)

    assert [r.message for r in report.of_kind(MISSED)] == ["gave up"]
    assert report.of_kind(OPTIMIZED)[0].helped


# how a kind is spelled once it reaches the output


def test_gcc_kinds_are_printed_bare() -> None:
    record = OptRecord(kind=MISSED, message="gave up", file="sum.c", line=3)
    assert kind_label(record) == MISSED


def test_a_named_pass_is_printed_with_the_kind() -> None:
    record = OptRecord(
        kind=MISSED, message="gave up", file="sum.c", line=3, pass_name="loop-vectorize"
    )
    # "missed" says something gave up, the pass name says what
    assert kind_label(record) == "missed (loop-vectorize)"


# and against whatever is really installed


@needs_compiler
def test_the_real_compiler_answers_one_way_or_the_other(sum_c: Path) -> None:
    compiler = find_compilers()[0]

    report = collect_report(sum_c, "2", compiler)

    assert report.via in FLAGS
    assert report.compiler == compiler
    # a summable loop at -O2 gives either compiler something to say
    assert report.records


@needs_compiler
def test_the_real_records_all_carry_a_kind_we_know(sum_c: Path) -> None:
    report = collect_report(sum_c, "2", find_compilers()[0])

    assert all(r.kind in (OPTIMIZED, MISSED, NOTE) for r in report.records)
    assert all(r.file.endswith("sum.c") for r in report.records)
