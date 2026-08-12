"""Every detector, and the two things you can do with the set of them.

A detector reads the asm for one function, decides that (say) the stack frame
is gone, and hands back an `Annotation` naming that along with the lines it
applies to. One optimization per module, with its own name and description
kept beside the code that finds it; `parsing.py` underneath holds the reading
they have in common.

Every detector takes the asm for one function — what `asm.isolate_function`
hands back, not a whole translation unit. The ones looking for something
visible in the optimized build take one body; the ones looking for something
that stopped being there take two, lowest level first.

This file is where they get collected: `find_annotations` runs the lot over a
pair of bodies, and `DESCRIPTIONS` is the same set keyed by name so `--explain`
can answer about an optimization nobody's file happened to contain.
"""

from compopt.annotation import Annotation
from compopt.detectors.arithmetic import (
    STRENGTH_REDUCTION,
    STRENGTH_REDUCTION_DESCRIPTION,
    detect_strength_reduction,
)
from compopt.detectors.branches import (
    BRANCH_ELIMINATION,
    BRANCH_ELIMINATION_DESCRIPTION,
    detect_branch_elimination,
    has_conditional_branch,
)
from compopt.detectors.calls import (
    INLINING,
    INLINING_DESCRIPTION,
    TAIL_CALL,
    TAIL_CALL_DESCRIPTION,
    called_functions,
    detect_inlining,
    detect_tail_call,
    inlined_callees,
    tail_jump_line,
)
from compopt.detectors.deadcode import (
    DEAD_CODE_ELIMINATION,
    DEAD_CODE_ELIMINATION_DESCRIPTION,
    detect_dead_code_elimination,
    instruction_count,
)
from compopt.detectors.folding import (
    CONSTANT_FOLDING,
    CONSTANT_FOLDING_DESCRIPTION,
    detect_constant_folding,
)
from compopt.detectors.frame import (
    FRAME_ELIMINATION,
    FRAME_ELIMINATION_DESCRIPTION,
    detect_frame_elimination,
    has_frame_setup,
)
from compopt.detectors.loops import (
    LOOP_UNROLLING,
    LOOP_UNROLLING_DESCRIPTION,
    detect_loop_unrolling,
    has_loop_branch,
)
from compopt.detectors.registers import (
    REGISTER_COALESCING,
    REGISTER_COALESCING_DESCRIPTION,
    detect_register_coalescing,
    uses_stack_slots,
)
from compopt.detectors.vectors import (
    VECTORIZATION,
    VECTORIZATION_DESCRIPTION,
    detect_vectorization,
    vector_line_range,
)

# the detectors that can work off the optimized body on its own, because the
# thing they look for is visible in it — a prologue that isn't there, a literal
# where a calculation used to be.
SINGLE_BODY_DETECTORS = (
    detect_frame_elimination,
    detect_constant_folding,
    detect_register_coalescing,
    detect_tail_call,
    detect_vectorization,
)

# and the ones that have to see -O0 as well, because what they look for is
# something that stopped being there and absence has no shape of its own.
#
# unrolling comes before branch elimination on purpose. both of them fire on a
# body that lost its jumps, and a loop written out in full is the more specific
# thing to say about one, so it gets asked first and `_drop_superseded` keeps
# the other quiet.
PAIRED_DETECTORS = (
    detect_dead_code_elimination,
    detect_loop_unrolling,
    detect_branch_elimination,
    detect_strength_reduction,
    detect_inlining,
)


# a finding that would only be repeating one already made, keyed by the name
# that wins. unrolling a loop removes its branch, so a body that reports both
# is describing one thing the compiler did and naming it twice; the specific
# one is the one worth keeping.
SUPERSEDED = {LOOP_UNROLLING: BRANCH_ELIMINATION}


# every optimization this knows how to name, and what it means. the detectors
# already carry the description onto each annotation they build; this is the
# same text reachable by name, so `--explain` can answer for an optimization
# that wasn't found in the file you happened to point at.
DESCRIPTIONS = {
    FRAME_ELIMINATION: FRAME_ELIMINATION_DESCRIPTION,
    CONSTANT_FOLDING: CONSTANT_FOLDING_DESCRIPTION,
    REGISTER_COALESCING: REGISTER_COALESCING_DESCRIPTION,
    DEAD_CODE_ELIMINATION: DEAD_CODE_ELIMINATION_DESCRIPTION,
    LOOP_UNROLLING: LOOP_UNROLLING_DESCRIPTION,
    TAIL_CALL: TAIL_CALL_DESCRIPTION,
    INLINING: INLINING_DESCRIPTION,
    STRENGTH_REDUCTION: STRENGTH_REDUCTION_DESCRIPTION,
    BRANCH_ELIMINATION: BRANCH_ELIMINATION_DESCRIPTION,
    VECTORIZATION: VECTORIZATION_DESCRIPTION,
}


def _drop_superseded(found: list[Annotation]) -> list[Annotation]:
    """Take out findings that another finding already accounts for."""
    names = {note.name for note in found}
    covered = {SUPERSEDED[name] for name in names if name in SUPERSEDED}
    return [note for note in found if note.name not in covered]


def find_annotations(baseline: str, optimized: str) -> list[Annotation]:
    """Run every detector over one function and collect what they found.

    Sorted by where they land rather than by the order the detectors happen to
    be listed in, so the notes come out in the order you meet them reading down
    the asm. Ties go to the shorter span, which puts the note about a single
    instruction above the one covering the whole body it sits in.

    Overlapping findings mostly stay, because they do overlap and both are
    usually true: a body small enough to keep everything in registers has
    normally lost instructions too, and coalescing and dead code are separate
    things the compiler did. The exception is a finding that is only another
    one restated, which `SUPERSEDED` lists and `_drop_superseded` removes.
    """
    found = [
        note
        for detect in SINGLE_BODY_DETECTORS
        if (note := detect(optimized)) is not None
    ]
    found += [
        note
        for detect in PAIRED_DETECTORS
        if (note := detect(baseline, optimized)) is not None
    ]
    return sorted(_drop_superseded(found), key=lambda note: (note.start, note.span, note.name))


def _normalize_name(name: str) -> str:
    """A name reduced to the bit that matters for matching it.

    The names have spaces in them, which means quoting them on a command line,
    so hyphens and underscores are taken as spaces too and `--explain
    dead-code-elimination` works without the quotes.
    """
    return " ".join(name.lower().replace("-", " ").replace("_", " ").split())


def match_name(name: str) -> str | None:
    """The optimization someone meant, spelled the way we spell it.

    An exact name wins. Failing that a unique prefix is enough, so `inlining`
    and `tail` both land — but a prefix matching two of them comes back as None
    rather than picking one, since guessing which was meant is worse than
    saying it was ambiguous.
    """
    wanted = _normalize_name(name)
    for known in DESCRIPTIONS:
        if _normalize_name(known) == wanted:
            return known

    hits = [known for known in DESCRIPTIONS if _normalize_name(known).startswith(wanted)]
    return hits[0] if len(hits) == 1 else None


def explain(name: str) -> str | None:
    """The description of an optimization by name, or None for an unknown one."""
    known = match_name(name)
    return None if known is None else DESCRIPTIONS[known]


# the names above are the package's public face — the detectors themselves, the
# few helpers worth calling on their own, and one constant per optimization.
# Listed out so it's obvious what's meant to be imported from here and what's
# an implementation detail of one of the modules.
__all__ = [
    "BRANCH_ELIMINATION",
    "CONSTANT_FOLDING",
    "DEAD_CODE_ELIMINATION",
    "DESCRIPTIONS",
    "FRAME_ELIMINATION",
    "INLINING",
    "LOOP_UNROLLING",
    "PAIRED_DETECTORS",
    "REGISTER_COALESCING",
    "SINGLE_BODY_DETECTORS",
    "STRENGTH_REDUCTION",
    "SUPERSEDED",
    "TAIL_CALL",
    "VECTORIZATION",
    "called_functions",
    "detect_branch_elimination",
    "detect_constant_folding",
    "detect_dead_code_elimination",
    "detect_frame_elimination",
    "detect_inlining",
    "detect_loop_unrolling",
    "detect_register_coalescing",
    "detect_strength_reduction",
    "detect_tail_call",
    "detect_vectorization",
    "explain",
    "find_annotations",
    "has_conditional_branch",
    "has_frame_setup",
    "has_loop_branch",
    "inlined_callees",
    "instruction_count",
    "match_name",
    "tail_jump_line",
    "uses_stack_slots",
    "vector_line_range",
]
