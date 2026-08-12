"""Dead code elimination: work the optimizer left out because nothing needed it."""

from compopt.annotation import Annotation
from compopt.detectors.frame import has_frame_setup
from compopt.detectors.parsing import instruction_line_range, parse_instruction

DEAD_CODE_ELIMINATION = "dead code elimination"
DEAD_CODE_ELIMINATION_DESCRIPTION = (
    "the optimized build reaches the same answer with fewer instructions, so "
    "the compiler found work in there that nothing depended on and left it out"
)

# push the base pointer, aim it at the stack, pop it back off at the end. that
# is the whole cost of a frame, and `frame.detect_frame_elimination` already
# reports it, so those three don't count as dead code when they go.
FRAME_INSTRUCTIONS = 3


def instruction_count(asm: str) -> int:
    """How many real instructions the body has in it.

    Labels, directives, comments and blanks all get dropped, so two bodies
    that came out of different compilers still compare fairly — gcc pads its
    output with a lot more .cfi bookkeeping than clang does, and counting
    lines instead of instructions would make that look like a difference in
    the code.
    """
    return sum(1 for line in asm.splitlines() if parse_instruction(line) is not None)


def _frame_allowance(baseline: str, optimized: str) -> int:
    """Instructions the missing prologue already accounts for.

    Only when the frame was there before and isn't now, which is the case
    `frame.detect_frame_elimination` fires on. If both builds have a frame, or
    neither does, nothing has to be set aside.
    """
    if has_frame_setup(baseline) and not has_frame_setup(optimized):
        return FRAME_INSTRUCTIONS
    return 0


def detect_dead_code_elimination(baseline: str, optimized: str) -> Annotation | None:
    """Spot work the optimizer dropped because nothing depended on it.

    This one needs two bodies, unlike the single-body detectors. Those all
    work off a shape you can see in the asm — a prologue that isn't there, a
    literal where a calculation used to be. Dead code doesn't leave a shape:
    the instructions are simply not written, and a body that never had any
    dead code in it looks exactly the same. So the only way to see it is to
    count what the unoptimized build did and compare.

    Three instructions of the drop are set aside first, because a function
    that loses its frame loses a push, a move and a pop without any of its
    work going away, and that's frame elimination's finding rather than this
    one's. Whatever is left over is code the compiler decided nothing needed.

    That still can't separate work that was deleted from work that was done
    in fewer instructions — the -O0 build of `a + b` spends six instructions
    going through memory where -O2 spends one lea, and none of that is dead.
    Same trade the other detectors make: the count says something went, not
    why. Reading what gcc reports for itself under -fopt-info would settle it,
    which is the next thing worth doing here.
    """
    if not baseline.strip() or not optimized.strip():
        # a missing side isn't a shrunken function, it's the function being
        # gone altogether — `diff.missing_message` is the one that reports it
        return None

    gone = instruction_count(baseline) - instruction_count(optimized)
    if gone - _frame_allowance(baseline, optimized) <= 0:
        return None

    span = instruction_line_range(optimized)
    if span is None:
        return None

    start, end = span
    return Annotation(DEAD_CODE_ELIMINATION, start, end, DEAD_CODE_ELIMINATION_DESCRIPTION)
