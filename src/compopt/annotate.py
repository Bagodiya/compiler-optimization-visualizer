"""The annotate command and the Annotation type it hands around.

Everything in the annotation engine ends up as an Annotation. A detector reads
the asm for a function, decides that (say) the stack frame is gone, and hands
back an Annotation naming that along with the lines it applies to. The
renderer later prints them next to those lines.

So far this is the shape they all share, the command wiring, and the
detectors written up to now; the rest of them get added underneath.
"""

from dataclasses import dataclass
from pathlib import Path

import typer

# The same prologue gets written two different ways depending on the syntax:
#
#     AT&T     pushq %rbp   then   movq %rsp, %rbp
#     Intel    push  rbp    then   mov  rbp, rsp
#
# rather than handle every spelling separately, the parsing below lowercases
# everything and drops the AT&T '%' so both flavours come out looking the same.
# The size suffixes are just listed out — there are only a couple that show up
# on these two instructions, and that's easier to read than stripping letters
# off the end of a mnemonic and hoping nothing else matches.
PUSH_MNEMONICS = ("push", "pushq", "pushl")
MOVE_MNEMONICS = ("mov", "movq", "movl")

BASE_POINTERS = ("rbp", "ebp")
STACK_POINTERS = ("rsp", "esp")

# whichever width the function returns, the value comes back in some slice of
# the same register, so a byte-sized return lands in al and an int in eax.
RETURN_REGISTERS = ("rax", "eax", "ax", "al")

# the move that drops a literal into that register — one spelling per width.
LOAD_MNEMONICS = ("mov", "movb", "movw", "movl", "movq")

# instructions that mean the machine is still working something out at run
# time. only the stems are listed: the sized spellings (addl, subq, imull...)
# come off the same stem, and there are far too many combinations to write out
# the way the two prologue mnemonics above are.
ARITHMETIC_STEMS = frozenset(
    {
        "add",
        "sub",
        "mul",
        "imul",
        "div",
        "idiv",
        "neg",
        "not",
        "inc",
        "dec",
        "and",
        "or",
        "xor",
        "shl",
        "sal",
        "shr",
        "sar",
        "lea",
    }
)
SIZE_SUFFIXES = ("b", "w", "l", "q")

FRAME_ELIMINATION = "stack frame elimination"
FRAME_ELIMINATION_DESCRIPTION = (
    "the prologue that saves the caller's base pointer and aims it at the stack "
    "is gone, so the function runs straight off the stack pointer and keeps its "
    "locals in registers"
)

CONSTANT_FOLDING = "constant folding"
CONSTANT_FOLDING_DESCRIPTION = (
    "the arithmetic was done by the compiler instead of by the program, so the "
    "function hands back a finished number and never computes anything"
)


@dataclass(frozen=True, slots=True)
class Annotation:
    """A named optimization, tied to the lines of asm that show it.

    The fields are:

    - ``name`` a short label for the optimization ("constant folding")
    - ``start`` and ``end`` the lines it covers, 1-based and inclusive. They
      are counted the same way `render.line_number_gutter` counts, so a
      number here matches what the reader sees in the left margin.
    - ``description`` a sentence saying what the compiler actually did, shown
      by ``--explain`` later on

    Frozen because a detector is finished with an annotation the moment it
    returns one — nothing downstream has any business renaming it or sliding
    the line range around.
    """

    name: str
    start: int
    end: int
    description: str = ""

    def __post_init__(self) -> None:
        # A bad range is a bug in whichever detector built this, and left
        # alone it turns into an annotation pointing at the wrong
        # instruction, which is a confusing thing to debug from the output.
        # Cheaper to complain at the point it was built.
        if self.start < 1:
            raise ValueError(f"start must be 1 or greater, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) comes before start ({self.start})")
        if not self.name.strip():
            raise ValueError("an annotation needs a name")

    @property
    def span(self) -> int:
        """How many lines the annotation covers — always at least one."""
        return self.end - self.start + 1

    def covers(self, line: int) -> bool:
        """True when the (1-based) `line` falls inside the range.

        This is what the renderer asks as it walks down the asm deciding
        which rows get a note beside them.
        """
        return self.start <= line <= self.end

    def label(self) -> str:
        """One-line form: the name plus the lines it applies to.

        Most annotations land on a single instruction, and "line 4" reads a
        lot better than "lines 4-4" for those, so the singular case gets its
        own wording.
        """
        where = f"line {self.start}" if self.span == 1 else f"lines {self.start}-{self.end}"
        return f"{self.name} ({where})"


def _parse_instruction(line: str) -> tuple[str, list[str]] | None:
    """Split one asm line into its mnemonic and its operands.

    Anything that isn't an instruction comes back as None — labels,
    directives, comment lines and blanks all land in the same bucket so the
    callers can skip them in one go instead of testing each case. The operands
    are lowercased and stripped of the AT&T '%' sigil, which is what makes the
    two syntaxes comparable at all.
    """
    code = line.split("#", 1)[0].strip()
    if not code or code.startswith(".") or code.endswith(":"):
        return None

    pieces = code.split(None, 1)
    mnemonic = pieces[0].lower()
    if len(pieces) == 1:
        # no-operand instruction, e.g. plain `ret`
        return mnemonic, []
    operands = [part.strip().lstrip("%").lower() for part in pieces[1].split(",")]
    return mnemonic, operands


def _aims_base_at_stack(operands: list[str]) -> bool:
    """True for the move that makes the base pointer the frame pointer.

    AT&T puts the source first (`movq %rsp, %rbp`) and Intel puts the
    destination first (`mov rbp, rsp`), so instead of picking a side we just
    check that the two operands are a stack pointer and a base pointer in
    whichever order they showed up. No other instruction in a prologue pairs
    those two registers, so there's nothing else this can accidentally match.
    """
    if len(operands) != 2:
        return False
    return any(op in BASE_POINTERS for op in operands) and any(
        op in STACK_POINTERS for op in operands
    )


def has_frame_setup(asm: str) -> bool:
    """True when the function still sets up a frame pointer on entry.

    Both halves have to be there. The push on its own only says the base
    pointer was saved, which an optimized build does whenever it wants rbp as
    one more scratch register — that's a callee-saved register, not a frame.
    It only becomes a frame once the move points it at the stack.

    They don't have to be next to each other: gcc slips its .cfi bookkeeping
    in between, and stripped or not, `_parse_instruction` skips over that.
    """
    saved_base = False
    for line in asm.splitlines():
        parsed = _parse_instruction(line)
        if parsed is None:
            continue
        mnemonic, operands = parsed
        if mnemonic in PUSH_MNEMONICS and operands and operands[0] in BASE_POINTERS:
            saved_base = True
        elif saved_base and mnemonic in MOVE_MNEMONICS and _aims_base_at_stack(operands):
            return True
    return False


def _first_instruction_line(asm: str) -> int | None:
    """Which line the function's first real instruction sits on, or None.

    Counted from 1 over the lines it was handed, so the number matches the
    gutter `render.line_number_gutter` draws for the same body.
    """
    for number, line in enumerate(asm.splitlines(), start=1):
        if _parse_instruction(line) is not None:
            return number
    return None


def detect_frame_elimination(asm: str) -> Annotation | None:
    """Spot a function that was compiled without a frame pointer.

    At -O0 every function opens by saving the caller's base pointer and
    pointing it at the stack, and then addresses all of its locals relative to
    it. Turn the optimizer on and that goes: rbp is an ordinary register
    again, the locals move into registers or get addressed off rsp, and the
    function loses an instruction at each end.

    There's no instruction to match on here, because the optimization *is* the
    missing prologue, so the annotation goes on the function's first
    instruction — the line the prologue used to occupy. Returns None when the
    frame is still there, and for a body with no instructions in it, since
    nothing was eliminated from a function that has no code.

    Takes the asm for a single function (what `asm.isolate_function` returns),
    not a whole translation unit.
    """
    if has_frame_setup(asm):
        return None

    line = _first_instruction_line(asm)
    if line is None:
        return None

    return Annotation(FRAME_ELIMINATION, line, line, FRAME_ELIMINATION_DESCRIPTION)


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


def _stem(mnemonic: str) -> str:
    """A mnemonic with its size suffix off: `addl` and `addq` both give `add`.

    Only suffixes that leave something recognisable behind get dropped, so
    `xorps` stays whole instead of turning into an integer xor.
    """
    if mnemonic not in ARITHMETIC_STEMS and mnemonic.endswith(SIZE_SUFFIXES):
        trimmed = mnemonic[:-1]
        if trimmed in ARITHMETIC_STEMS:
            return trimmed
    return mnemonic


def _does_arithmetic(mnemonic: str) -> bool:
    """True for an instruction that works a value out at run time."""
    return _stem(mnemonic) in ARITHMETIC_STEMS


def _loads_a_literal(mnemonic: str, operands: list[str]) -> bool:
    """True for an instruction that puts a constant in the return register.

    The two operands get checked without caring which is which, for the same
    reason `_aims_base_at_stack` does: AT&T and Intel disagree about the
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
    if _stem(mnemonic) == "xor":
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
        parsed = _parse_instruction(line)
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

    Takes the asm for a single function, same as the other detectors.
    """
    line = _literal_return_line(asm)
    if line is None:
        return None

    return Annotation(CONSTANT_FOLDING, line, line, CONSTANT_FOLDING_DESCRIPTION)


def run_annotate(path: Path) -> None:
    """Entry point for `compopt annotate`.

    Eventually this compiles the file and points out the optimizations the
    compiler applied, one Annotation per thing it spotted. Right now it only
    checks the file and says what it's going to do, so the detectors get
    added to a command that already exists and is wired into the CLI.
    """
    if not path.exists():
        # same as show/diff: a plain error line beats a traceback
        typer.echo(f"error: no such file: {path}", err=True)
        raise typer.Exit(code=1)

    if not path.is_file():
        typer.echo(f"error: not a file: {path}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"would annotate the optimizations in {path}")
