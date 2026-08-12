"""Vectorization: work moved into the wide registers, several values at a time."""

from compopt.annotation import Annotation
from compopt.detectors.parsing import parse_instruction

# the wide registers. only the prefix is listed: the numbering runs to xmm15 on
# its own and further with the wider forms, and none of that changes the answer.
VECTOR_REGISTERS = ("xmm", "ymm", "zmm")

# floating point comes in two flavours out of the same registers. `ss` and `sd`
# are one value, `ps` and `pd` are a register full of them, and that suffix is
# the only thing separating ordinary double arithmetic from vector work.
SCALAR_SUFFIXES = ("ss", "sd")
PACKED_SUFFIXES = ("ps", "pd", "dq", "dqa", "dqu")

# integer vector instructions are marked on the front instead — paddd, pmulld,
# pcmpeqb. nothing scalar starts with a p, so the prefix is enough on its own.
PACKED_PREFIX = "p"

VECTORIZATION = "vectorization"
VECTORIZATION_DESCRIPTION = (
    "the work is being done on several values at a time in the wide registers "
    "rather than one per trip round the loop, so each instruction here is "
    "doing what took four or eight of them before"
)


def _is_vector_instruction(mnemonic: str, operands: list[str]) -> bool:
    """True for an instruction working on several values at once.

    The register isn't enough on its own. Scalar floating point lives in the
    same xmm registers that vector code uses — `addss` adds one float and
    `addps` adds four — so a body doing ordinary `double` arithmetic would read
    as vectorized if naming an xmm register were the test.

    What separates them is the suffix: `ps` and `pd` are packed, `ss` and `sd`
    are scalar. Integer vector work is marked differently again, with a `p` on
    the front (`paddd`, `pmulld`), and the moves that load a whole register at
    a time (`movaps`, `movdqu`) are packed by definition.
    """
    if any(operand.startswith(VECTOR_REGISTERS) for operand in operands):
        if mnemonic.endswith(SCALAR_SUFFIXES):
            return False
        if mnemonic.endswith(PACKED_SUFFIXES) or mnemonic.startswith(PACKED_PREFIX):
            return True
    return False


def vector_line_range(asm: str) -> tuple[int, int] | None:
    """First and last line doing vector work, or None when there is none."""
    first = None
    last = None
    for number, line in enumerate(asm.splitlines(), start=1):
        parsed = parse_instruction(line)
        if parsed is None:
            continue
        if _is_vector_instruction(*parsed):
            if first is None:
                first = number
            last = number
    if first is None or last is None:
        return None
    return first, last


def detect_vectorization(asm: str) -> Annotation | None:
    """Spot a loop the compiler rewrote to work on several values at a time.

    The registers are wide enough to hold more than one number — an xmm holds
    four ints, a ymm holds eight — and there are instructions that operate on
    all of them at once. So a loop adding up an array doesn't have to go round
    once per element; the compiler can have it do four or eight per trip and
    add the pieces together at the end.

    One body is enough, which most of the two-body detectors can't say. C has
    no way to ask for a packed add, so a packed instruction in the output is
    always something the compiler decided, never something that was written.

    What this can't tell you is how wide the vectorization went or how much of
    the loop it covered — that needs the compiler's own account of it, which
    gcc gives under -fopt-info-vec.
    """
    span = vector_line_range(asm)
    if span is None:
        return None

    start, end = span
    return Annotation(VECTORIZATION, start, end, VECTORIZATION_DESCRIPTION)
