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

# a local that didn't get a register lives at some offset from one of those
# two, and both syntaxes wrap the address in brackets of their own:
#
#     AT&T     -4(%rbp)          movl -4(%rbp), %eax
#     Intel    [rbp - 4]         mov  eax, dword ptr [rbp - 4]
#
# so a stack slot is a bracket of either kind with a frame register inside it.
FRAME_REGISTERS = BASE_POINTERS + STACK_POINTERS
MEMORY_BRACKETS = ("(", "[")

# whichever width the function returns, the value comes back in some slice of
# the same register, so a byte-sized return lands in al and an int in eax.
RETURN_REGISTERS = ("rax", "eax", "ax", "al")

# the move that drops a literal into that register — one spelling per width.
LOAD_MNEMONICS = ("mov", "movb", "movw", "movl", "movq")

# every jump in the instruction set starts with a j — jmp, je, jne, jle, and
# the rest of them — and nothing else does, so one letter is enough here where
# the other groups needed a list. the loop family is the exception: it counts
# rcx down and jumps in the same instruction, and the name doesn't start with
# a j, so those get written out.
JUMP_PREFIX = "j"
COUNTING_JUMPS = ("loop", "loope", "loopz", "loopne", "loopnz")

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

REGISTER_COALESCING = "register coalescing"
REGISTER_COALESCING_DESCRIPTION = (
    "the locals were given registers to live in rather than a stack slot each, "
    "so the function works on its values in place instead of writing them out "
    "to memory and loading them back for every step"
)

DEAD_CODE_ELIMINATION = "dead code elimination"
DEAD_CODE_ELIMINATION_DESCRIPTION = (
    "the optimized build reaches the same answer with fewer instructions, so "
    "the compiler found work in there that nothing depended on and left it out"
)

LOOP_UNROLLING = "loop unrolling"
LOOP_UNROLLING_DESCRIPTION = (
    "the body of the loop was written out several times over instead of being "
    "jumped back to, so the work runs in a straight line with no counter to "
    "keep and no branch to take on every trip"
)

# push the base pointer, aim it at the stack, pop it back off at the end. that
# is the whole cost of a frame, and `detect_frame_elimination` already reports
# it, so those three don't count as dead code when they go.
FRAME_INSTRUCTIONS = 3

# how long a stretch has to be before we call it a repeat. two instructions of
# the same kind next to each other is what ordinary code looks like — a pair of
# moves setting up a call, say — and three is where it starts to look like a
# copy of something. the cost of the floor is that a two-instruction body
# unrolled twice reads as ordinary code and gets missed, which is the better
# way round to be wrong: a missed unroll is quiet, a wrong one is misleading.
MINIMUM_REPEATED_INSTRUCTIONS = 3


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


def _instruction_line_range(asm: str) -> tuple[int, int] | None:
    """The first and last lines holding an instruction, or None for neither.

    Counted from 1 over the lines it was handed, so the numbers match the
    gutter `render.line_number_gutter` draws for the same body. The ends are
    both instructions, which means the label on top and any trailing .size
    directive stay outside the range — an annotation covering those would be
    pointing at lines that aren't code.
    """
    first = None
    last = None
    for number, line in enumerate(asm.splitlines(), start=1):
        if _parse_instruction(line) is None:
            continue
        if first is None:
            first = number
        last = number
    if first is None or last is None:
        return None
    return first, last


def _first_instruction_line(asm: str) -> int | None:
    """Which line the function's first real instruction sits on, or None."""
    span = _instruction_line_range(asm)
    return None if span is None else span[0]


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
        parsed = _parse_instruction(line)
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
    `detect_constant_folding` makes: one body can't tell you what the -O0
    build looked like.

    The annotation covers the whole body rather than a single line, since it's
    the shape of all the instructions together that shows it, not any one of
    them. None comes back for a body with nothing in it to keep in registers.

    Takes the asm for a single function, same as the other detectors.
    """
    if uses_stack_slots(asm):
        return None

    span = _instruction_line_range(asm)
    if span is None:
        return None

    start, end = span
    return Annotation(REGISTER_COALESCING, start, end, REGISTER_COALESCING_DESCRIPTION)


def instruction_count(asm: str) -> int:
    """How many real instructions the body has in it.

    Labels, directives, comments and blanks all get dropped, so two bodies
    that came out of different compilers still compare fairly — gcc pads its
    output with a lot more .cfi bookkeeping than clang does, and counting
    lines instead of instructions would make that look like a difference in
    the code.
    """
    return sum(1 for line in asm.splitlines() if _parse_instruction(line) is not None)


def _frame_allowance(baseline: str, optimized: str) -> int:
    """Instructions the missing prologue already accounts for.

    Only when the frame was there before and isn't now, which is the case
    `detect_frame_elimination` fires on. If both builds have a frame, or
    neither does, nothing has to be set aside.
    """
    if has_frame_setup(baseline) and not has_frame_setup(optimized):
        return FRAME_INSTRUCTIONS
    return 0


def detect_dead_code_elimination(baseline: str, optimized: str) -> Annotation | None:
    """Spot work the optimizer dropped because nothing depended on it.

    This one needs two bodies, unlike the detectors above it. Those all work
    off a shape you can see in the asm — a prologue that isn't there, a
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
    why. Cross-referencing gcc's own pass output settles it, which is what
    Phase 5 is for.

    Takes the asm for one function at each level, low first.
    """
    if not baseline.strip() or not optimized.strip():
        # a missing side isn't a shrunken function, it's the function being
        # gone altogether — `diff.missing_message` is the one that reports it
        return None

    gone = instruction_count(baseline) - instruction_count(optimized)
    if gone - _frame_allowance(baseline, optimized) <= 0:
        return None

    span = _instruction_line_range(optimized)
    if span is None:
        return None

    start, end = span
    return Annotation(DEAD_CODE_ELIMINATION, start, end, DEAD_CODE_ELIMINATION_DESCRIPTION)


def _label_name(line: str) -> str | None:
    """The label a line defines, lowercased, or None when it defines none.

    `_parse_instruction` throws these away with the directives, since a label
    isn't code, but a jump target is a label and this is the one place that
    has to look at them. Lowercased for the same reason the operands are: it's
    the only way `.L2` and the `.l2` an operand comes back as compare equal.
    """
    text = line.split("#", 1)[0].strip()
    if not text.endswith(":"):
        return None
    name = text.removesuffix(":").strip().lower()
    return name or None


def _jump_target(mnemonic: str, operands: list[str]) -> str | None:
    """The label a jump goes to, or None when the line isn't a plain jump.

    An indirect jump through a register (`jmp *%rax`, from a switch table)
    has an operand that was never a label, so it falls out here on its own
    without a special case — nothing it could name is in the label set.
    """
    if len(operands) != 1:
        return None
    if mnemonic.startswith(JUMP_PREFIX) or mnemonic in COUNTING_JUMPS:
        return operands[0]
    return None


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
        label = _label_name(line)
        if label is not None:
            passed.add(label)
            continue
        parsed = _parse_instruction(line)
        if parsed is None:
            continue
        mnemonic, operands = parsed
        target = _jump_target(mnemonic, operands)
        if target is not None and target in passed:
            return True
    return False


def _instruction_mnemonics(asm: str) -> list[tuple[int, str]]:
    """Every instruction in the body as its line number and its mnemonic."""
    found = []
    for number, line in enumerate(asm.splitlines(), start=1):
        parsed = _parse_instruction(line)
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

    Two bodies, for the same reason `detect_dead_code_elimination` needs two.
    A straight-line run of similar instructions on its own says nothing — the
    source may well have been written that way. It only means unrolling if
    there was a loop there to begin with, and that's in the -O0 build.

    So there are three things to see: a backwards jump in the baseline, no
    backwards jump left in the optimized build, and a repeated block in what
    replaced it. The last one is what separates unrolling from the loop being
    turned into a closed form — `sum_to_n` at -O2 comes out as a multiply and
    a shift, which also has no branch left but isn't the body written out
    again.

    Only fully unrolled loops, then. Partial unrolling — four copies per trip
    with the loop still going round — keeps its backwards jump and reads as
    an ordinary loop here. Catching those means knowing how many copies the
    baseline had per trip, which the two bodies don't say; gcc's own pass
    output does, and that's what Phase 5 is about.

    Takes the asm for one function at each level, low first.
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
