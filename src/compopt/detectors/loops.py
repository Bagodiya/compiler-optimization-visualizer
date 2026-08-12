"""Loop unrolling: the loop the compiler replaced with copies of its own body."""

from compopt.annotation import Annotation
from compopt.detectors.parsing import jump_target, label_name, parse_instruction

LOOP_UNROLLING = "loop unrolling"
LOOP_UNROLLING_DESCRIPTION = (
    "the body of the loop was written out several times over instead of being "
    "jumped back to, so the work runs in a straight line with no counter to "
    "keep and no branch to take on every trip"
)

# how long a stretch has to be before we call it a repeat. two instructions of
# the same kind next to each other is what ordinary code looks like — a pair of
# moves setting up a call, say — and three is where it starts to look like a
# copy of something. the cost of the floor is that a two-instruction body
# unrolled twice reads as ordinary code and gets missed, which is the better
# way round to be wrong: a missed unroll is quiet, a wrong one is misleading.
MINIMUM_REPEATED_INSTRUCTIONS = 3


def has_loop_branch(asm: str) -> bool:
    """True when some jump in the body goes back to a label above it.

    That backwards jump is what a loop is once it's been compiled: the body
    runs, the counter gets checked and the jump sends control back up to do it
    again. A forward jump is an if/else stepping over the arm it didn't take,
    which is why the direction matters and the presence of a jump doesn't.

    Deciding the direction is just bookkeeping — walk down the body keeping
    the labels already passed, and a jump naming one of those is going
    backwards by definition.
    """
    passed: set[str] = set()
    for line in asm.splitlines():
        label = label_name(line)
        if label is not None:
            passed.add(label)
            continue
        parsed = parse_instruction(line)
        if parsed is None:
            continue
        mnemonic, operands = parsed
        target = jump_target(mnemonic, operands)
        if target is not None and target in passed:
            return True
    return False


def _instruction_mnemonics(asm: str) -> list[tuple[int, str]]:
    """Every instruction in the body as its line number and its mnemonic."""
    found = []
    for number, line in enumerate(asm.splitlines(), start=1):
        parsed = parse_instruction(line)
        if parsed is not None:
            found.append((number, parsed[0]))
    return found


def _repeated_run(mnemonics: list[str]) -> tuple[int, int] | None:
    """The longest stretch that is one block of mnemonics written out again.

    Comes back as the first and last index into `mnemonics`, or None when
    nothing in there repeats. Only the mnemonics are compared and not the
    operands, because the copies of an unrolled body aren't identical: each
    one reaches a different element, so the four adds of a summing loop come
    out as `addl (%rdi)`, `addl 4(%rdi)`, `addl 8(%rdi)` and so on. The
    operations are the repeated part, the addresses are what's different
    about each copy.

    Every block length gets tried, shortest first, because the block that
    repeats is the loop body and there's no telling how long that was. A
    stretch has to hold at least two copies to count, and be long enough to
    clear MINIMUM_REPEATED_INSTRUCTIONS.
    """
    best: tuple[int, int] | None = None
    best_length = 0
    for period in range(1, len(mnemonics) // 2 + 1):
        start = 0
        while start + period < len(mnemonics):
            # push `end` out for as long as each mnemonic matches the one a
            # block behind it, which is what makes the stretch a repetition
            end = start + period
            while end < len(mnemonics) and mnemonics[end] == mnemonics[end - period]:
                end += 1
            length = end - start
            repeats = length >= 2 * period and length >= MINIMUM_REPEATED_INSTRUCTIONS
            if repeats and length > best_length:
                best, best_length = (start, end - 1), length
            # the next stretch worth trying starts one past where this block
            # last lined up; anything earlier is inside the run just measured
            start = end - period + 1
    return best


def detect_loop_unrolling(baseline: str, optimized: str) -> Annotation | None:
    """Spot a loop the compiler replaced with copies of its own body.

    A loop costs more than the work inside it: something has to count the
    trips, compare, and jump back for the next one. When the compiler can
    work out how many trips there will be, it can pay that once by writing
    the body out one copy per trip, and the counter and the branch both go.

    Two bodies, for the same reason `deadcode.detect_dead_code_elimination`
    needs two. A straight-line run of similar instructions on its own says
    nothing — the source may well have been written that way. It only means
    unrolling if there was a loop there to begin with, and that's in the -O0
    build.

    So there are three things to see: a backwards jump in the baseline, no
    backwards jump left in the optimized build, and a repeated block in what
    replaced it. The last one is what separates unrolling from the loop being
    turned into a closed form — `sum_to_n` at -O2 comes out as a multiply and
    a shift, which also has no branch left but isn't the body written out
    again.

    Only fully unrolled loops, then. Partial unrolling — four copies per trip
    with the loop still going round — keeps its backwards jump and reads as
    an ordinary loop here. Catching those means knowing how many copies the
    baseline had per trip, which the two bodies don't say; gcc says it under
    -fopt-info-loop, which is the way in if this needs to get smarter.
    """
    if not baseline.strip() or not optimized.strip():
        return None

    if not has_loop_branch(baseline) or has_loop_branch(optimized):
        return None

    instructions = _instruction_mnemonics(optimized)
    run = _repeated_run([mnemonic for _, mnemonic in instructions])
    if run is None:
        return None

    first, last = run
    start = instructions[first][0]
    end = instructions[last][0]
    return Annotation(LOOP_UNROLLING, start, end, LOOP_UNROLLING_DESCRIPTION)
