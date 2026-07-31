"""Tests for the annotate command, the Annotation type, and the detectors.

Some of these build annotations by hand and check the range bookkeeping — the
part every detector depends on. The rest feed hand-written function bodies to
the detectors, one body per case, so it's obvious from the sample what's meant
to be found. The command itself is still a skeleton, so those tests only check
it validates the file it was handed.
"""

import dataclasses
from pathlib import Path

import pytest
from typer.testing import CliRunner

from compopt.annotate import (
    CONSTANT_FOLDING,
    FRAME_ELIMINATION,
    REGISTER_COALESCING,
    Annotation,
    detect_constant_folding,
    detect_frame_elimination,
    detect_register_coalescing,
    has_frame_setup,
    uses_stack_slots,
)
from compopt.cli import app

runner = CliRunner()

# what -O0 looks like: rbp is saved, aimed at the stack, and every local is
# addressed off it. the .cfi line is in there on purpose — it sits between the
# two prologue instructions in real gcc output.
FRAMED = """add:
.LFB0:
\t.cfi_startproc
\tpushq\t%rbp
\t.cfi_def_cfa_offset 16
\tmovq\t%rsp, %rbp
\tmovl\t%edi, -4(%rbp)
\tmovl\t%esi, -8(%rbp)
\tmovl\t-4(%rbp), %eax
\taddl\t-8(%rbp), %eax
\tpopq\t%rbp
\tret
"""

# the same function at -O2: no prologue at all, the arguments never touch memory
FLAT = """add:
\tleal\t(%rdi,%rsi), %eax
\tret
"""

# clang with -masm=intel writes the prologue the other way round
INTEL_FRAMED = """add:
\tpush\trbp
\tmov\trbp, rsp
\tmov\tdword ptr [rbp - 4], edi
\tpop\trbp
\tret
"""

# rbp gets pushed here, but only because the function wants it as a spare
# register across the call — there's no move aiming it at the stack, so this
# is a callee-saved register and not a frame
SAVES_RBP = """work:
\tpushq\t%rbp
\tpushq\t%rbx
\tmovq\t%rdi, %rbx
\tcall\thelper
\tpopq\t%rbx
\tpopq\t%rbp
\tret
"""

# examples/const_fold.c at -O0: 7*6 is folded on its own, but the rest of the
# arithmetic is still there, each step going out to a local and back
UNFOLDED = """compute:
\tpushq\t%rbp
\tmovq\t%rsp, %rbp
\tmovl\t$42, -4(%rbp)
\tmovl\t-4(%rbp), %eax
\taddl\t$100, %eax
\tmovl\t%eax, -8(%rbp)
\tmovl\t-8(%rbp), %eax
\tsubl\t$42, %eax
\tshll\t$1, %eax
\tpopq\t%rbp
\tret
"""

# the same function at -O2, with the whole sum collapsed into one number
FOLDED = """compute:
\tpushq\t%rbp
\tmovq\t%rsp, %rbp
\tmovl\t$200, %eax
\tpopq\t%rbp
\tret
"""

INTEL_FOLDED = """compute:
\tmov\teax, 200
\tret
"""

# `return 0` — the xor is shorter to encode than moving a zero in, so that's
# what both compilers emit
ZERO_RETURN = """main:
\txorl\t%eax, %eax
\tret
"""

# the return value comes from the caller, so there was nothing to fold
IDENTITY = """identity:
\tmovl\t%edi, %eax
\tret
"""

# a literal, but it goes to a global instead of the return register
STORES_LITERAL = """set_flag:
\tmovl\t$1, flag(%rip)
\tret
"""

# folded as far as it goes, then multiplied by an argument at run time
HALF_FOLDED = """scale:
\tmovl\t$200, %eax
\timull\t%edi, %eax
\tret
"""

# the frame pointer is gone but the locals are still in memory, addressed off
# rsp now instead of rbp — this is the case that frame elimination alone would
# get wrong if we treated the missing prologue as proof the values are in
# registers
SPILLS_TO_RSP = """work:
\tsubq\t$24, %rsp
\tmovl\t%edi, 12(%rsp)
\tcall\thelper
\taddl\t12(%rsp), %eax
\taddq\t$24, %rsp
\tret
"""

# everything happens between registers, and rbp is only pushed because the
# function borrows it as a spare — no offset off it anywhere
KEEPS_IN_REGISTERS = """work:
\tpushq\t%rbp
\tmovl\t%edi, %ebp
\tcall\thelper
\taddl\t%ebp, %eax
\tpopq\t%rbp
\tret
"""

# a global, which lives in memory no matter what the allocator does, so it
# shouldn't read as a spilled local
READS_GLOBAL = """get_flag:
\tmovl\tflag(%rip), %eax
\tret
"""

INTEL_READS_GLOBAL = """get_flag:
\tmov\teax, dword ptr [rip + flag]
\tret
"""


def test_fields_round_trip() -> None:
    note = Annotation("constant folding", 3, 5, "the arithmetic was done at compile time")
    assert note.name == "constant folding"
    assert note.start == 3
    assert note.end == 5
    assert note.description == "the arithmetic was done at compile time"


def test_description_is_optional() -> None:
    note = Annotation("dead code elimination", 1, 2)
    assert note.description == ""


def test_span_counts_both_ends() -> None:
    assert Annotation("inlining", 4, 4).span == 1
    assert Annotation("inlining", 4, 9).span == 6


def test_covers_is_inclusive() -> None:
    note = Annotation("loop unrolling", 10, 14)
    assert note.covers(10)
    assert note.covers(12)
    assert note.covers(14)
    assert not note.covers(9)
    assert not note.covers(15)


def test_label_uses_singular_for_one_line() -> None:
    assert Annotation("tail call", 7, 7).label() == "tail call (line 7)"


def test_label_uses_a_range_for_several_lines() -> None:
    assert Annotation("vectorization", 7, 11).label() == "vectorization (lines 7-11)"


def test_start_below_one_is_rejected() -> None:
    # line numbers start at 1 in the gutter, so 0 means someone passed a
    # list index straight through without adjusting it
    with pytest.raises(ValueError, match="start must be 1 or greater"):
        Annotation("branch elimination", 0, 3)


def test_backwards_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="comes before start"):
        Annotation("strength reduction", 8, 5)


def test_blank_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="needs a name"):
        Annotation("   ", 1, 2)


def test_annotation_is_frozen() -> None:
    note = Annotation("register coalescing", 2, 3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        note.name = "something else"


def test_frame_setup_found_in_att_prologue() -> None:
    assert has_frame_setup(FRAMED)


def test_frame_setup_found_in_intel_prologue() -> None:
    assert has_frame_setup(INTEL_FRAMED)


def test_no_frame_setup_without_a_prologue() -> None:
    assert not has_frame_setup(FLAT)


def test_pushing_rbp_alone_is_not_a_frame() -> None:
    assert not has_frame_setup(SAVES_RBP)


def test_framed_function_gets_no_annotation() -> None:
    assert detect_frame_elimination(FRAMED) is None
    assert detect_frame_elimination(INTEL_FRAMED) is None


def test_flat_function_is_annotated() -> None:
    note = detect_frame_elimination(FLAT)
    assert note is not None
    assert note.name == FRAME_ELIMINATION
    assert note.description


def test_annotation_points_at_the_first_instruction() -> None:
    # line 1 is the "add:" label, so the lea on line 2 is where the prologue
    # would have been
    note = detect_frame_elimination(FLAT)
    assert note is not None
    assert (note.start, note.end) == (2, 2)


def test_annotation_skips_labels_and_directives() -> None:
    # same body as FLAT with a pile of noise on top; the annotation should
    # still land on the instruction, not on the first line of the block
    padded = "add:\n.LFB0:\n\t.cfi_startproc\n\tleal\t(%rdi,%rsi), %eax\n\tret\n"
    note = detect_frame_elimination(padded)
    assert note is not None
    assert note.start == 4


def test_callee_saved_rbp_still_counts_as_eliminated() -> None:
    note = detect_frame_elimination(SAVES_RBP)
    assert note is not None
    assert note.start == 2


def test_empty_body_gives_nothing() -> None:
    assert detect_frame_elimination("") is None


def test_body_with_no_instructions_gives_nothing() -> None:
    # a label and a directive on their own aren't a function that lost a frame
    assert detect_frame_elimination("add:\n\t.cfi_startproc\n") is None


def test_folded_function_is_annotated() -> None:
    note = detect_constant_folding(FOLDED)
    assert note is not None
    assert note.name == CONSTANT_FOLDING
    assert note.description


def test_folding_points_at_the_literal() -> None:
    # line 4 is the `movl $200, %eax` that replaced the whole calculation
    note = detect_constant_folding(FOLDED)
    assert note is not None
    assert (note.start, note.end) == (4, 4)


def test_folding_found_in_intel_syntax() -> None:
    note = detect_constant_folding(INTEL_FOLDED)
    assert note is not None
    assert note.start == 2


def test_zeroing_the_return_register_counts() -> None:
    note = detect_constant_folding(ZERO_RETURN)
    assert note is not None
    assert note.start == 2


def test_remaining_arithmetic_means_no_folding() -> None:
    assert detect_constant_folding(UNFOLDED) is None
    assert detect_constant_folding(HALF_FOLDED) is None


def test_returning_an_argument_is_not_folding() -> None:
    assert detect_constant_folding(IDENTITY) is None


def test_literal_stored_elsewhere_is_not_folding() -> None:
    assert detect_constant_folding(STORES_LITERAL) is None


def test_empty_body_has_nothing_to_fold() -> None:
    assert detect_constant_folding("") is None
    assert detect_constant_folding("compute:\n\t.cfi_startproc\n") is None


def test_stack_slots_found_off_rbp() -> None:
    assert uses_stack_slots(FRAMED)


def test_stack_slots_found_off_rsp() -> None:
    assert uses_stack_slots(SPILLS_TO_RSP)


def test_stack_slots_found_in_intel_syntax() -> None:
    assert uses_stack_slots(INTEL_FRAMED)


def test_no_stack_slots_when_everything_is_in_registers() -> None:
    assert not uses_stack_slots(FLAT)
    assert not uses_stack_slots(KEEPS_IN_REGISTERS)


def test_pushing_rbp_is_not_a_stack_slot() -> None:
    # the register is named, but nothing is addressed off it
    assert not uses_stack_slots(SAVES_RBP)


def test_globals_are_not_stack_slots() -> None:
    assert not uses_stack_slots(READS_GLOBAL)
    assert not uses_stack_slots(INTEL_READS_GLOBAL)
    assert not uses_stack_slots(STORES_LITERAL)


def test_register_only_body_is_annotated() -> None:
    note = detect_register_coalescing(KEEPS_IN_REGISTERS)
    assert note is not None
    assert note.name == REGISTER_COALESCING
    assert note.description


def test_coalescing_covers_the_whole_body() -> None:
    # line 1 is the label, so the range runs from the push on line 2 to the
    # ret on line 7
    note = detect_register_coalescing(KEEPS_IN_REGISTERS)
    assert note is not None
    assert (note.start, note.end) == (2, 7)


def test_spilled_locals_are_not_annotated() -> None:
    assert detect_register_coalescing(FRAMED) is None
    assert detect_register_coalescing(INTEL_FRAMED) is None


def test_missing_prologue_alone_is_not_enough() -> None:
    # the frame pointer is gone, so frame elimination fires here, but the
    # locals are still going out to memory
    assert detect_frame_elimination(SPILLS_TO_RSP) is not None
    assert detect_register_coalescing(SPILLS_TO_RSP) is None


def test_coalescing_found_in_intel_syntax() -> None:
    note = detect_register_coalescing(INTEL_FOLDED)
    assert note is not None
    assert (note.start, note.end) == (2, 3)


def test_empty_body_has_nothing_in_registers() -> None:
    assert detect_register_coalescing("") is None
    assert detect_register_coalescing("work:\n\t.cfi_startproc\n") is None


def test_annotate_names_the_file(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    result = runner.invoke(app, ["annotate", str(src)])
    assert result.exit_code == 0
    # the placeholder should at least say which file it would work on
    assert "hello.c" in result.stdout


def test_annotate_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.c"

    result = runner.invoke(app, ["annotate", str(missing)])
    assert result.exit_code == 1


def test_annotate_directory_is_rejected(tmp_path: Path) -> None:
    # a directory isn't a source file, so this fails like a missing one
    result = runner.invoke(app, ["annotate", str(tmp_path)])
    assert result.exit_code == 1


def test_annotate_shows_up_in_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "annotate" in result.stdout
