"""Tests for stripping noisy assembler directives."""

from compopt.asm import strip_directives

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
