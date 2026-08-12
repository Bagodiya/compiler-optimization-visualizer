"""Strength reduction: a multiply or a divide traded for something cheaper."""

from compopt.annotation import Annotation
from compopt.detectors.parsing import instruction_line_range, parse_instruction, stem

# the instructions worth going out of your way to avoid. a multiply takes a few
# cycles and a divide takes tens of them, against one for a shift, which is the
# whole reason the compiler trades them away.
EXPENSIVE_STEMS = frozenset({"mul", "imul", "div", "idiv"})

# and what it trades them for. lea is in here because it adds and shifts in one
# instruction without touching the flags, which makes it the usual landing
# place for a multiply by a constant that isn't a power of two.
CHEAP_STEMS = frozenset({"shl", "sal", "shr", "sar", "lea", "add", "sub"})

STRENGTH_REDUCTION = "strength reduction"
STRENGTH_REDUCTION_DESCRIPTION = (
    "a multiply or a divide was swapped for shifts and adds that reach the "
    "same answer, because the operand turned out to be something the cheap "
    "instructions could handle"
)


def _counts_instructions(asm: str, wanted: frozenset[str]) -> int:
    """How many instructions in the body have a stem in `wanted`."""
    total = 0
    for line in asm.splitlines():
        parsed = parse_instruction(line)
        if parsed is not None and stem(parsed[0]) in wanted:
            total += 1
    return total


def detect_strength_reduction(baseline: str, optimized: str) -> Annotation | None:
    """Spot a multiply or divide the compiler swapped for something cheaper.

    A multiply costs several cycles and a divide costs tens of them, while a
    shift costs one. When the operand is a power of two the compiler doesn't
    need the expensive instruction at all: `x * 8` is `x << 3`, and `x / 4` on
    an unsigned value is `x >> 2`. Multiplying by a nearby constant goes the
    same way — `x * 5` comes out as an lea adding x to itself shifted twice.

    Two bodies, because a shift on its own says nothing: `x << 3` is something
    C can say directly, and code that was written that way looks identical to
    code that was reduced into it. What makes it the compiler's doing is a
    multiply or divide that was in the -O0 build and isn't here any more.

    The cheap instructions have to have actually turned up, which is what
    separates this from the multiply simply being deleted — that's dead code,
    and `deadcode.detect_dead_code_elimination` is the one that reports it.

    In practice this fires less often than you'd expect, and the reason is
    worth knowing: clang does the easy reductions while it's choosing
    instructions rather than while it's optimizing, so `x * 8` is already a
    shift at -O0 and there was never a multiply for us to see go. What's left
    to catch is the kind that needs real analysis — a multiply inside a loop
    turned into an add carried between trips — which happens at -O1 and above
    where the baseline genuinely has the expensive instruction in it.
    """
    if not baseline.strip() or not optimized.strip():
        return None

    before = _counts_instructions(baseline, EXPENSIVE_STEMS)
    after = _counts_instructions(optimized, EXPENSIVE_STEMS)
    if before == 0 or after >= before:
        return None

    if _counts_instructions(optimized, CHEAP_STEMS) == 0:
        return None

    span = instruction_line_range(optimized)
    if span is None:
        return None

    start, end = span
    return Annotation(STRENGTH_REDUCTION, start, end, STRENGTH_REDUCTION_DESCRIPTION)
