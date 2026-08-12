"""Constant folding: the arithmetic the compiler did so the program doesn't."""

from compopt.annotation import Annotation
from compopt.detectors.parsing import ARITHMETIC_STEMS, parse_instruction, stem

# whichever width the function returns, the value comes back in some slice of
# the same register, so a byte-sized return lands in al and an int in eax.
RETURN_REGISTERS = ("rax", "eax", "ax", "al")

# the move that drops a literal into that register — one spelling per width.
LOAD_MNEMONICS = ("mov", "movb", "movw", "movl", "movq")

CONSTANT_FOLDING = "constant folding"
CONSTANT_FOLDING_DESCRIPTION = (
    "the arithmetic was done by the compiler instead of by the program, so the "
    "function hands back a finished number and never computes anything"
)


def _immediate_value(operand: str) -> int | None:
    """The number an operand holds, or None when it isn't a literal.

    AT&T marks immediates with a '$' and Intel just writes the number, so the
    sigil comes off first and both spellings parse the same way after that.
    Base 0 is deliberate — it takes the 0x form gcc uses for larger constants
    as well as plain decimal.
    """
    try:
        return int(operand.removeprefix("$"), 0)
    except ValueError:
        return None


def _does_arithmetic(mnemonic: str) -> bool:
    """True for an instruction that works a value out at run time."""
    return stem(mnemonic) in ARITHMETIC_STEMS


def _loads_a_literal(mnemonic: str, operands: list[str]) -> bool:
    """True for an instruction that puts a constant in the return register.

    The two operands get checked without caring which is which, for the same
    reason `frame._aims_base_at_stack` does: AT&T and Intel disagree about the
    order, and no other instruction pairs the return register with a number.
    """
    if len(operands) != 2:
        return False

    if mnemonic in LOAD_MNEMONICS:
        return any(op in RETURN_REGISTERS for op in operands) and any(
            _immediate_value(op) is not None for op in operands
        )

    # both compilers write `return 0` as `xor %eax, %eax` rather than moving a
    # zero in, because the xor encodes shorter. it's a constant either way.
    if stem(mnemonic) == "xor":
        return operands[0] == operands[1] and operands[0] in RETURN_REGISTERS

    return False


def _literal_return_line(asm: str) -> int | None:
    """Where the return value gets its literal, if the body has no maths left.

    One pass does both halves of the question. A literal load is remembered
    and the walk carries on, because the interesting part is what comes after
    it; the first arithmetic instruction anywhere in the body ends the search,
    since something is still being computed and the fold wasn't complete.
    """
    found = None
    for number, line in enumerate(asm.splitlines(), start=1):
        parsed = parse_instruction(line)
        if parsed is None:
            continue
        mnemonic, operands = parsed
        if _loads_a_literal(mnemonic, operands):
            # keep the earliest one: with several returns in a body they are
            # all folded, and the first is the one the reader meets first
            found = number if found is None else found
        elif _does_arithmetic(mnemonic):
            return None
    return found


def detect_constant_folding(asm: str) -> Annotation | None:
    """Spot a function whose arithmetic was done at compile time.

    `int c = 7 * 6 + 100 - 42; return c * 2;` is a multiply, an add, a
    subtract and a shift at -O0, each one writing its result to a local and
    reading it back. The optimizer can see that none of it depends on
    anything the caller passes in, works the answer out itself, and emits
    `movl $200, %eax`. So the tell is a literal going into the return
    register with no arithmetic anywhere around it.

    The catch is that a function which was written as `return 200;` compiles
    to exactly that same instruction, and nothing in the asm says whether the
    constant was in the source or worked out by the compiler. Comparing
    against the -O0 build would settle it, and the diff command already
    knows how to line two levels up, but a detector only gets one body — so
    this reports the shape it sees and takes the false positive.
    """
    line = _literal_return_line(asm)
    if line is None:
        return None

    return Annotation(CONSTANT_FOLDING, line, line, CONSTANT_FOLDING_DESCRIPTION)
