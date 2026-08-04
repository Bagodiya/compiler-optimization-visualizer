"""Tests for stripping noisy assembler directives."""

import pytest

from compopt.asm import (
    find_function,
    function_names,
    isolate_function,
    strip_directives,
)

# a small but realistic chunk of gcc output with the usual noise around it
SAMPLE = """\t.file\t"add.c"
\t.text
\t.globl\tadd
\t.type\tadd, @function
add:
.LFB0:
\t.cfi_startproc
\tpushq\t%rbp
\t.cfi_def_cfa_offset 16
\tmovq\t%rsp, %rbp
\taddl\t%esi, %eax
\tpopq\t%rbp
\tret
\t.cfi_endproc
.LFE0:
\t.size\tadd, .-add
\t.ident\t"GCC: (Debian) 12.2.0"
\t.section\t.note.GNU-stack,"",@progbits
"""


def test_drops_known_noise_directives() -> None:
    out = strip_directives(SAMPLE)
    for directive in (".file", ".cfi_", ".size", ".ident", ".section", ".type"):
        assert directive not in out


def test_keeps_instructions_and_labels() -> None:
    out = strip_directives(SAMPLE)
    # the actual code and the function label/.L labels must survive
    assert "add:" in out
    assert ".LFB0:" in out
    assert "addl\t%esi, %eax" in out
    assert "ret" in out


def test_keeps_indentation_of_kept_lines() -> None:
    out = strip_directives("\tpushq\t%rbp\n")
    assert out == "\tpushq\t%rbp"


def test_empty_input_gives_empty_output() -> None:
    assert strip_directives("") == ""


# two functions back to back, already cleaned of directives
TWO_FUNCS = """add:
.LFB0:
\taddl\t%esi, %edi
\tret
sub:
.LFB1:
\tsubl\t%esi, %edi
\tret"""


def test_function_names_lists_top_level_labels() -> None:
    assert function_names(TWO_FUNCS) == ["add", "sub"]


def test_function_names_skips_local_labels() -> None:
    # the .L labels are the compiler's own bookkeeping, not functions
    assert ".LFB0" not in function_names(TWO_FUNCS)


def test_isolate_first_function_by_default() -> None:
    out = isolate_function(TWO_FUNCS)
    assert out.startswith("add:")
    assert "addl\t%esi, %edi" in out
    # must not bleed into the next function
    assert "sub:" not in out


def test_isolate_named_function() -> None:
    out = isolate_function(TWO_FUNCS, "sub")
    assert out.startswith("sub:")
    assert "subl\t%esi, %edi" in out
    assert "add:" not in out


def test_isolate_unknown_function_raises() -> None:
    with pytest.raises(KeyError):
        isolate_function(TWO_FUNCS, "nope")


def test_isolate_returns_empty_when_no_functions() -> None:
    assert isolate_function("\tnop\n\tret") == ""


def test_find_function_matches_isolate_when_the_function_is_there() -> None:
    assert find_function(TWO_FUNCS, "sub") == isolate_function(TWO_FUNCS, "sub")


def test_find_function_gives_empty_instead_of_raising() -> None:
    # this is the whole reason it exists — a function that got inlined away
    # at a higher -O level is a result, not a crash
    assert find_function(TWO_FUNCS, "nope") == ""


def test_find_function_still_defaults_to_the_first_one() -> None:
    assert find_function(TWO_FUNCS).startswith("add:")


# Mach-O local labels sit at column 0, which is where a function label goes too

MACHO = """_sum:                                   ## @sum
## %bb.0:
\ttestl\t%edi, %edi
\tjle\tLBB0_1
\tmovl\t%edi, %eax
\tretq
LBB0_1:
\txorl\t%eax, %eax
\tretq
"""


def test_clang_local_labels_are_not_functions() -> None:
    assert function_names(MACHO) == ["_sum"]


def test_a_function_is_not_cut_off_at_its_own_local_label() -> None:
    # isolate_function stops at the next function label, so counting LBB0_1 as
    # one would drop the whole branch that follows it
    body = isolate_function(MACHO)
    assert "LBB0_1:" in body
    assert body.strip().endswith("retq")
    assert len(body.splitlines()) == len(MACHO.rstrip().splitlines())



def test_the_other_mach_o_labels_are_skipped_too() -> None:
    # constant pools, jump tables, temporaries, string literals, and the
    # frame markers GNU gcc wraps every function in
    asm = ("_f:\n\tretq\nLCPI0_0:\nLJTI0_0:\nLtmp3:\nL_.str:\n"
           "LFB0:\nLFE0:\nEH_frame1:\n")
    assert function_names(asm) == ["_f"]


def test_gnu_gcc_frame_labels_do_not_end_a_function() -> None:
    # GNU gcc targeting Darwin brackets each function with LFB0/LFE0, and the
    # body sits between them — stopping at LFB0 would leave nothing at all
    asm = "_add:\nLFB0:\n\tleal\t(%rdi,%rsi), %eax\n\tret\nLFE0:\n"
    body = isolate_function(asm)
    assert "leal" in body
    assert "ret" in body


def test_a_function_can_still_start_with_an_l_on_elf() -> None:
    # on ELF a leading L means nothing special, so a real function named this
    # way has to survive
    assert function_names("Lookup:\n\tret\n") == ["Lookup"]


def test_elf_is_not_mistaken_for_mach_o_by_one_underscore() -> None:
    # a Linux build can have an underscored symbol in it without that making
    # every other function invisible
    elf = "_start:\n\tret\nmain:\n.L2:\n\tret\n"
    assert function_names(elf) == ["_start", "main"]


# what gcc leaves behind on ELF for a file with nothing in it. the numeric
# labels are the assembler's own, referred to as `1f`/`1b` rather than by name
GNU_PROPERTY_NOTE = """\t.section\t.note.gnu.property,"a"
\t.align 8
\t.long\t 1f - 0f
\t.long\t 4f - 1f
\t.long\t 5
0:
\t.string\t "GNU"
1:
\t.align 8
4:
"""


def test_numeric_labels_are_not_functions() -> None:
    assert function_names(GNU_PROPERTY_NOTE) == []


def test_a_file_with_no_functions_isolates_to_nothing() -> None:
    # this is what `annotate` leans on to say there was nothing to look at;
    # picking `0:` up as a function made it report the note block instead
    assert isolate_function(strip_directives(GNU_PROPERTY_NOTE)) == ""


def test_numeric_labels_do_not_end_a_real_function() -> None:
    # the note block sits in the same translation unit as real code
    asm = "add:\n\tleal\t(%rdi,%rsi), %eax\n\tret\n" + GNU_PROPERTY_NOTE
    assert function_names(asm) == ["add"]
    assert "leal" in isolate_function(asm)
