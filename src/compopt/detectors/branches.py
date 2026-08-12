"""Branch elimination: the decision the compiler made so the processor needn't."""

from compopt.annotation import Annotation
from compopt.detectors.parsing import (
    COUNTING_JUMPS,
    JUMP_PREFIX,
    UNCONDITIONAL_JUMPS,
    instruction_line_range,
    parse_instruction,
)

BRANCH_ELIMINATION = "branch elimination"
BRANCH_ELIMINATION_DESCRIPTION = (
    "the condition no longer decides where to go — either the compiler worked "
    "out which way it went and kept that arm, or it computed both sides and "
    "picked the answer with a conditional move, so nothing can mispredict"
)


def _is_conditional_jump(mnemonic: str) -> bool:
    """True for a jump that asks a question before it goes.

    Everything starting with a j is a jump and only `jmp` goes every time, so
    the two spellings of that one come out and the rest are conditional. The
    counting jumps belong here too: `loop` decrements rcx and only jumps while
    it's non-zero, which is a condition however it's spelled.
    """
    if mnemonic in COUNTING_JUMPS:
        return True
    return mnemonic.startswith(JUMP_PREFIX) and mnemonic not in UNCONDITIONAL_JUMPS


def has_conditional_branch(asm: str) -> bool:
    """True when anything in the body decides where to go next."""
    for line in asm.splitlines():
        parsed = parse_instruction(line)
        if parsed is None:
            continue
        if _is_conditional_jump(parsed[0]):
            return True
    return False


def detect_branch_elimination(baseline: str, optimized: str) -> Annotation | None:
    """Spot a decision the compiler worked out so the branch didn't have to.

    A branch that the processor guesses wrong costs more than the work on
    either side of it, so the compiler would rather not have one. Sometimes it
    can prove which way the condition goes and keeps only that arm. Other times
    it runs both sides and picks the answer at the end with a conditional move,
    which has no branch in it at all — `a > b ? a : b` becomes a compare and a
    `cmovg` rather than a jump over the arm not taken.

    Two bodies again. A body with no branches in it usually never had any, and
    the -O0 build is the only thing that says otherwise.

    A loop that got unrolled also loses its branch, and this would report that
    as well if it were asked, so `loops.detect_loop_unrolling` is checked first
    and the ordering in `PAIRED_DETECTORS` is what keeps them apart.
    """
    if not baseline.strip() or not optimized.strip():
        return None

    if not has_conditional_branch(baseline) or has_conditional_branch(optimized):
        return None

    span = instruction_line_range(optimized)
    if span is None:
        return None

    start, end = span
    return Annotation(BRANCH_ELIMINATION, start, end, BRANCH_ELIMINATION_DESCRIPTION)
