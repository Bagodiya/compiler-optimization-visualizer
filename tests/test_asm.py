"""Tests for stripping noisy assembler directives."""

import pytest

from compopt.asm import function_names, isolate_function, strip_directives

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
