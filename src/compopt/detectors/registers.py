"""Register coalescing: the locals that never had to go out to the stack."""

from compopt.annotation import Annotation
from compopt.detectors.parsing import (
    FRAME_REGISTERS,
    MEMORY_BRACKETS,
    instruction_line_range,
    parse_instruction,
)

REGISTER_COALESCING = "register coalescing"
REGISTER_COALESCING_DESCRIPTION = (
    "the locals were given registers to live in rather than a stack slot each, "
    "so the function works on its values in place instead of writing them out "
    "to memory and loading them back for every step"
)


def _is_stack_slot(operand: str) -> bool:
    """True when the operand is memory addressed off rbp or rsp.

    A frame register on its own isn't enough — `pushq %rbp` names rbp and
    doesn't touch a local. It has to be inside brackets for the operand to be
    an address rather than the register's own value.

    Globals go through the same brackets but off a different register, so
    `flag(%rip)` and `[rip + flag]` don't match here, which is what we want:
    a global was always going to be in memory and no allocator was ever going
    to keep it somewhere else.
    """
    if not any(bracket in operand for bracket in MEMORY_BRACKETS):
        return False
    # the '%' is still in there on the AT&T side, since the parser only takes
    # the leading one off, but a plain substring test doesn't care either way
    return any(register in operand for register in FRAME_REGISTERS)


def uses_stack_slots(asm: str) -> bool:
    """True when anything in the body reads or writes a local in memory."""
    for line in asm.splitlines():
        parsed = parse_instruction(line)
        if parsed is None:
            continue
        _, operands = parsed
        if any(_is_stack_slot(operand) for operand in operands):
            return True
    return False


def detect_register_coalescing(asm: str) -> Annotation | None:
    """Spot a function whose locals never went out to the stack.

    At -O0 every variable gets its own slot in the frame and every use of it
    is a load and a store, so `a + b` reads both arguments back out of memory
    after they were just written there. With the optimizer on, the allocator
    hands the variables registers, and the ones that only pass a value along
    end up sharing a register instead of each having their own — that's the
    coalescing part. What you see in the asm is a body doing all of its work
    between registers with the frame slots gone.

    So the tell is the absence of any operand addressed off rbp or rsp. That
    also means a function which never had anything to spill — two arguments
    added and returned, say — reports as coalesced, because its asm looks
    exactly like a function whose spills were optimized away. Same trade as
    `folding.detect_constant_folding` makes: one body can't tell you what the
    -O0 build looked like.

    The annotation covers the whole body rather than a single line, since it's
    the shape of all the instructions together that shows it, not any one of
    them. None comes back for a body with nothing in it to keep in registers.
    """
    if uses_stack_slots(asm):
        return None

    span = instruction_line_range(asm)
    if span is None:
        return None

    start, end = span
    return Annotation(REGISTER_COALESCING, start, end, REGISTER_COALESCING_DESCRIPTION)
