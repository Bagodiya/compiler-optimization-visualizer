"""Tests for matching the compiler's pass report up with lines of assembly."""

from pathlib import Path

import pytest

from compopt.asm import isolate_function, strip_directives
from compopt.compilers import compile_to_asm, find_compilers
from compopt.crossref import (
    LocatedRecord,
    cross_reference,
    file_table,
    strip_debug_lines,
)
from compopt.report import MISSED, NOTE, OPTIMIZED, OptRecord

LOOP_C = """int total(const int *xs, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += xs[i];
    return sum;
}
"""

# cut down from `clang -S -g -O2` on the file above, with the run of vector
# loads left in because the interleaving is the whole difficulty: source lines
# 3 and 4 take turns the whole way down the unrolled body.
LOOP_ASM = """\t.file\t1 "/tmp" "loop.c"
_total:
\t.loc\t1 1 0
\tpushq\t%rbp
\t.loc\t1 3 23 prologue_end
\ttestl\t%esi, %esi
\t.loc\t1 0 5
\txorl\t%eax, %eax
LBB0_7:
\t##DEBUG_VALUE: total:sum <- 0
\t.loc\t1 4 16 is_stmt 1
\tmovdqu\t(%rdi,%rsi,4), %xmm2
\t.loc\t1 4 13 is_stmt 0
\tpaddd\t%xmm0, %xmm2
\t.loc\t1 3 29 is_stmt 1
\taddq\t$32, %rsi
\tjne\tLBB0_7
\t.loc\t1 5 5
\tretq
"""


def located(records, asm):
    """Run a whole asm body through the join, the way a caller would."""
    _, origin = strip_debug_lines(asm)
    return cross_reference(records, origin, file_table(asm))


# reading the file table


def test_reads_the_mach_o_two_string_form() -> None:
    assert file_table('\t.file\t1 "/tmp" "loop.c"\n') == {1: "loop.c"}


def test_reads_the_elf_one_string_form() -> None:
    assert file_table('\t.file\t1 "loop.c"\n') == {1: "loop.c"}


def test_the_unnumbered_file_directive_is_not_an_entry() -> None:
    # gcc writes this one at the top to name the primary source; .loc never
    # refers to it, and letting it in would need a number it hasn't got
    assert file_table('\t.file\t"loop.c"\n') == {}


def test_several_files_all_land_in_the_table() -> None:
    asm = '\t.file\t1 "loop.c"\n\t.file\t2 "/usr/include/stdio.h"\n'

    # kept as written: the ELF form has nowhere to put a directory, so the
    # path is the string, and it's the name that gets compared later anyway
    assert file_table(asm) == {1: "loop.c", 2: "/usr/include/stdio.h"}


# pulling the debug lines back out


def test_loc_directives_are_gone_from_the_body() -> None:
    body, _ = strip_debug_lines(LOOP_ASM)

    assert ".loc" not in body


def test_debug_value_comments_are_gone_too() -> None:
    body, _ = strip_debug_lines(LOOP_ASM)

    assert "DEBUG_VALUE" not in body


def test_the_instructions_are_all_still_there() -> None:
    body, _ = strip_debug_lines(LOOP_ASM)

    assert "\tpushq\t%rbp" in body
    assert "\tretq" in body
    assert "LBB0_7:" in body


def test_lines_are_numbered_after_the_directives_come_out() -> None:
    body, origin = strip_debug_lines("\t.loc\t1 3 5\n\tmovl\t$0, %eax\n")

    # one line left, so the instruction is line 1 and not line 2
    assert body == "\tmovl\t$0, %eax"
    assert origin == {1: (1, 3)}


def test_a_run_carries_on_until_the_next_loc() -> None:
    asm = "\t.loc\t1 4 16\n\taddl\t%eax, %ebx\n\taddl\t%ecx, %ebx\n\t.loc\t1 5 5\n\tretq\n"
    _, origin = strip_debug_lines(asm)

    assert origin == {1: (1, 4), 2: (1, 4), 3: (1, 5)}


def test_lines_before_the_first_loc_belong_to_nobody() -> None:
    _, origin = strip_debug_lines(LOOP_ASM)

    # the .file line and the _total: label come before anything is claimed
    assert 1 not in origin
    assert 2 not in origin


def test_line_zero_claims_nothing() -> None:
    # .loc 1 0 5 is the compiler saying this bit isn't from any line of C
    _, origin = strip_debug_lines("\t.loc\t1 0 5\n\txorl\t%eax, %eax\n")

    assert origin == {}


def test_line_zero_ends_the_run_before_it() -> None:
    asm = "\t.loc\t1 3 5\n\tmovl\t$0, %eax\n\t.loc\t1 0 5\n\txorl\t%eax, %eax\n"
    _, origin = strip_debug_lines(asm)

    # the xorl must not keep inheriting line 3 from further up
    assert origin == {1: (1, 3)}


def test_a_body_with_no_debug_info_comes_back_untouched() -> None:
    plain = "_add:\n\tmovl\t%edi, %eax\n\tretq"
    body, origin = strip_debug_lines(plain)

    assert body == plain
    assert origin == {}


def test_an_empty_body_is_fine() -> None:
    assert strip_debug_lines("") == ("", {})


# the join itself


def test_a_record_gets_the_lines_from_its_source_line() -> None:
    record = OptRecord(OPTIMIZED, "loop vectorized", "loop.c", 4)

    [found] = located([record], LOOP_ASM)

    body, _ = strip_debug_lines(LOOP_ASM)
    lines = body.splitlines()
    assert [lines[n - 1].strip() for n in found.lines] == [
        "movdqu\t(%rdi,%rsi,4), %xmm2",
        "paddd\t%xmm0, %xmm2",
    ]


def test_records_keep_the_order_they_were_reported_in() -> None:
    records = [
        OptRecord(NOTE, "considering", "loop.c", 5),
        OptRecord(OPTIMIZED, "vectorized", "loop.c", 4),
    ]

    assert [f.record.line for f in located(records, LOOP_ASM)] == [5, 4]


def test_a_source_line_with_no_asm_finds_nothing() -> None:
    # line 2 is `int sum = 0;`, which -O2 folds away and never emits
    record = OptRecord(MISSED, "gave up", "loop.c", 2)

    [found] = located([record], LOOP_ASM)

    assert not found.found
    assert found.lines == ()


def test_a_record_from_another_file_does_not_match() -> None:
    # same line number, different file — without the table this would land on
    # the vector loads and claim a header was what got optimized
    record = OptRecord(OPTIMIZED, "inlined", "stdio.h", 4)

    [found] = located([record], LOOP_ASM)

    assert not found.found


def test_the_file_is_matched_on_its_name_not_its_path() -> None:
    # gcc reports the path it was handed, the .file entry has the one clang
    # resolved, and they don't have to be spelled the same way
    record = OptRecord(OPTIMIZED, "vectorized", "/home/me/src/loop.c", 4)

    [found] = located([record], LOOP_ASM)

    assert found.found


def test_no_table_means_the_line_number_is_enough() -> None:
    _, origin = strip_debug_lines(LOOP_ASM)
    record = OptRecord(OPTIMIZED, "vectorized", "anything.c", 4)

    [found] = cross_reference([record], origin)

    assert found.found


def test_a_file_number_missing_from_the_table_matches_nothing() -> None:
    # .loc says file 2 and the table only knows file 1, so we can't tell what
    # that run is and won't guess
    asm = '\t.file\t1 "loop.c"\n\t.loc\t2 4 16\n\tmovl\t%eax, %ebx\n'
    record = OptRecord(OPTIMIZED, "vectorized", "loop.c", 4)

    [found] = located([record], asm)

    assert not found.found


def test_nothing_to_join_gives_nothing_back() -> None:
    assert located([], LOOP_ASM) == []


# the record type


def test_start_and_end_bracket_the_matched_lines() -> None:
    found = LocatedRecord(OptRecord(OPTIMIZED, "x", "loop.c", 4), (6, 7, 9))

    assert found.start == 6
    assert found.end == 9
    assert found.found


def test_nothing_matched_has_no_start_or_end() -> None:
    found = LocatedRecord(OptRecord(MISSED, "x", "loop.c", 4))

    assert found.start is None
    assert found.end is None
    assert not found.found


def test_neighbouring_lines_become_one_range() -> None:
    found = LocatedRecord(OptRecord(OPTIMIZED, "x", "loop.c", 4), (3, 4, 5))

    assert found.runs() == [(3, 5)]


def test_a_split_up_line_keeps_its_pieces_apart() -> None:
    # what an unrolled body looks like: line 4 in three places with line 3's
    # instructions in the gaps, and saying 3-12 would claim those too
    found = LocatedRecord(OptRecord(OPTIMIZED, "x", "loop.c", 4), (3, 4, 8, 11, 12))

    assert found.runs() == [(3, 4), (8, 8), (11, 12)]


def test_one_line_on_its_own_is_a_range_of_one() -> None:
    found = LocatedRecord(OptRecord(OPTIMIZED, "x", "loop.c", 4), (7,))

    assert found.runs() == [(7, 7)]


def test_nothing_matched_has_no_ranges() -> None:
    assert LocatedRecord(OptRecord(MISSED, "x", "loop.c", 4)).runs() == []


# against a compiler that's really installed


def debug_compiler() -> str | None:
    """Whichever installed compiler we can get .loc directives out of."""
    for name in find_compilers():
        if name == "clang" or name == "gcc":
            return name
    return None


def test_real_asm_has_a_file_table_and_loc_runs(tmp_path: Path) -> None:
    compiler = debug_compiler()
    if compiler is None:
        pytest.skip("no compiler on this machine to build with")

    src = tmp_path / "loop.c"
    src.write_text(LOOP_C)

    asm = compile_to_asm(src, "2", compiler, debug=True)
    table = file_table(asm)
    body = isolate_function(strip_directives(asm), "total")
    clean, origin = strip_debug_lines(body)

    assert "loop.c" in table.values()
    assert origin
    assert ".loc" not in clean


def test_the_loop_body_lands_on_real_instructions(tmp_path: Path) -> None:
    compiler = debug_compiler()
    if compiler is None:
        pytest.skip("no compiler on this machine to build with")

    src = tmp_path / "loop.c"
    src.write_text(LOOP_C)

    asm = compile_to_asm(src, "2", compiler, debug=True)
    body = isolate_function(strip_directives(asm), "total")
    clean, origin = strip_debug_lines(body)

    # line 4 is `sum += xs[i];`, so whatever it compiled to has to add
    record = OptRecord(OPTIMIZED, "loop vectorized", str(src), 4)
    [found] = cross_reference([record], origin, file_table(asm))

    lines = clean.splitlines()
    assert found.found
    assert any("add" in lines[n - 1] for n in found.lines)


def test_every_matched_line_is_inside_the_body(tmp_path: Path) -> None:
    compiler = debug_compiler()
    if compiler is None:
        pytest.skip("no compiler on this machine to build with")

    src = tmp_path / "loop.c"
    src.write_text(LOOP_C)

    asm = compile_to_asm(src, "2", compiler, debug=True)
    body = isolate_function(strip_directives(asm), "total")
    clean, origin = strip_debug_lines(body)

    records = [OptRecord(NOTE, "something", str(src), n) for n in range(1, 7)]
    count = len(clean.splitlines())

    # a number outside the body would point the renderer at a row that isn't
    # on screen, which is the failure this whole module exists to avoid
    for found in cross_reference(records, origin, file_table(asm)):
        assert all(1 <= n <= count for n in found.lines)


def instructions(body: str) -> list[str]:
    """Just the instruction lines — no labels, directives or comments."""
    kept = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith((".", "#")) or stripped.endswith(":"):
            continue
        kept.append(stripped)
    return kept


def test_debug_asm_and_plain_asm_hold_the_same_instructions(tmp_path: Path) -> None:
    compiler = debug_compiler()
    if compiler is None:
        pytest.skip("no compiler on this machine to build with")

    src = tmp_path / "loop.c"
    src.write_text(LOOP_C)

    with_debug = compile_to_asm(src, "2", compiler, debug=True)
    plain = compile_to_asm(src, "2", compiler)

    clean, _ = strip_debug_lines(isolate_function(strip_directives(with_debug), "total"))
    same = isolate_function(strip_directives(plain), "total")

    # the join is only worth anything if -g leaves the code alone, so this is
    # the claim the whole module rests on. Compared instruction by instruction
    # rather than line by line because -g really does add lines — its own
    # Lfunc_begin/Ltmp labels, and the debug tables at the end of the file,
    # which isolate_function sweeps up because there's no function label after
    # the last function to stop it. Those tables are all directives and so
    # don't show up here, but they're still in the body, and whatever displays
    # a -g build is going to have to cut them off first.
    assert instructions(clean) == instructions(same)
