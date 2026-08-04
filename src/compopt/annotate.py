"""The annotate command and the Annotation type it hands around.

Everything in the annotation engine ends up as an Annotation. A detector reads
the asm for a function, decides that (say) the stack frame is gone, and hands
back an Annotation naming that along with the lines it applies to. The
renderer prints them next to those lines.

Every detector takes the asm for one function — what `asm.isolate_function`
hands back, not a whole translation unit. The ones looking for something
visible in the optimized build take one body; the ones looking for something
that stopped being there take two, lowest level first.

The shape they all share comes first, then the detectors, then the command
that runs the lot of them over one function.
"""

from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console

from compopt.asm import find_function, function_names, isolate_function, strip_directives
from compopt.compilers import (
    check_level,
    choose_compiler,
    compile_at_levels,
    normalize_level,
)
from compopt.render import render_annotated

# the level everything is compared against. the detectors that need two bodies
# need one where the compiler hasn't done anything yet, and that's -O0.
BASELINE_LEVEL = "0"

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

# the two spellings of the jump that always goes. everything else beginning
# with a j is asking a question first, and a jump that might not happen leaves
# the function still running underneath it.
UNCONDITIONAL_JUMPS = ("jmp", "jmpq")

# a jump can also go through a register instead of naming anything, which is
# what a switch big enough to get a jump table compiles to. AT&T marks that
# with a '*' and Intel just writes the register bare, so the sigil doesn't
# catch both and the names have to be listed. Only the 64-bit ones: what a
# jump wants is an address, and an address here is 64 bits wide.
ADDRESS_REGISTERS = frozenset(
    {
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "rbp",
        "rsp",
        "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
    }
)

# the two spellings of a call. same story as the push and move mnemonics at the
# top of the file: only a couple of them ever turn up, so listing them reads
# better than taking letters off the end.
CALL_MNEMONICS = ("call", "callq")

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

# the instructions worth going out of your way to avoid. a multiply takes a few
# cycles and a divide takes tens of them, against one for a shift, which is the
# whole reason the compiler trades them away.
EXPENSIVE_STEMS = frozenset({"mul", "imul", "div", "idiv"})

# and what it trades them for. lea is in here because it adds and shifts in one
# instruction without touching the flags, which makes it the usual landing
# place for a multiply by a constant that isn't a power of two.
CHEAP_STEMS = frozenset({"shl", "sal", "shr", "sar", "lea", "add", "sub"})

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

TAIL_CALL = "tail call optimization"
TAIL_CALL_DESCRIPTION = (
    "the last thing the function does is hand over to another one, and it "
    "jumps there rather than calling it and waiting, so nothing comes back "
    "here — the callee borrows this frame and returns straight to our caller"
)

STRENGTH_REDUCTION = "strength reduction"
STRENGTH_REDUCTION_DESCRIPTION = (
    "a multiply or a divide was swapped for shifts and adds that reach the "
    "same answer, because the operand turned out to be something the cheap "
    "instructions could handle"
)

BRANCH_ELIMINATION = "branch elimination"
BRANCH_ELIMINATION_DESCRIPTION = (
    "the condition no longer decides where to go — either the compiler worked "
    "out which way it went and kept that arm, or it computed both sides and "
    "picked the answer with a conditional move, so nothing can mispredict"
)

VECTORIZATION = "vectorization"
VECTORIZATION_DESCRIPTION = (
    "the work is being done on several values at a time in the wide registers "
    "rather than one per trip round the loop, so each instruction here is "
    "doing what took four or eight of them before"
)

INLINING = "inlining"
INLINING_DESCRIPTION = (
    "a function this one used to call isn't called any more, because its body "
    "was copied in here instead — so the work happens in place, with no "
    "arguments to arrange, no jump away and no trip back"
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


def _local_labels(asm: str) -> set[str]:
    """Every label the body defines, lowercased.

    `has_loop_branch` collects these as it walks, because it cares about which
    ones it has already gone past. Here the direction doesn't matter — a jump
    naming any of them is staying inside this function, and a jump naming
    something else is leaving it, which is the whole question below.
    """
    labels: set[str] = set()
    for line in asm.splitlines():
        label = _label_name(line)
        if label is not None:
            labels.add(label)
    return labels


def _is_indirect(operand: str) -> bool:
    """True when a jump goes through a register or through memory.

    `jmp *%rax` and `jmp qword ptr [rax]` both land here. The address only
    exists once the program is running, so there's no name in the operand to
    compare against anything, and the table it came out of points back into
    this same function anyway.
    """
    if operand.startswith("*"):
        return True
    if any(bracket in operand for bracket in MEMORY_BRACKETS):
        return True
    return operand in ADDRESS_REGISTERS


def _final_instruction(asm: str) -> tuple[int, str, list[str]] | None:
    """The last instruction in the body: its line, mnemonic and operands.

    None when there are no instructions in there at all. Whatever the compiler
    leaves underneath doesn't count — gcc writes .cfi_endproc and a .size line
    after the last instruction and clang doesn't, and neither of them is code,
    so the function ends in the same place either way.
    """
    found = None
    for number, line in enumerate(asm.splitlines(), start=1):
        parsed = _parse_instruction(line)
        if parsed is not None:
            mnemonic, operands = parsed
            found = (number, mnemonic, operands)
    return found


def tail_jump_line(asm: str) -> int | None:
    """Which line jumps away to another function, when that's how the body ends.

    Three things have to hold, and each one rules out something that looks
    similar. The last instruction is an unconditional jump: a conditional one
    falls through to more of our own code, so the function isn't finished with
    it. The target is a name rather than a register, since an indirect jump is
    a jump table landing back inside here. And the name isn't a label this body
    defines, because a jump to one of those is a loop or an if.
    """
    final = _final_instruction(asm)
    if final is None:
        return None

    number, mnemonic, operands = final
    if mnemonic not in UNCONDITIONAL_JUMPS or len(operands) != 1:
        return None

    target = operands[0]
    if _is_indirect(target) or target in _local_labels(asm):
        return None
    return number


def detect_tail_call(asm: str) -> Annotation | None:
    """Spot a call at the end of a function that became a jump.

    `return helper(x);` at -O0 is a call and then a ret. helper gets a stack
    frame of its own, does its work, returns here, and this function turns
    round and passes the same value straight back up. That middle stop is
    wasted — there is nothing left to do here once helper is done. So the
    optimizer drops the ret and jumps to helper instead of calling it: helper
    reuses the frame that's already there and returns to our caller directly.

    Worth a frame and a return each time, and it's also what lets deep
    recursion finish. A tail-recursive function that jumps instead of calling
    stops stacking frames up, so it can't run the stack out.

    One body is enough here, which the last few detectors couldn't say. They
    need the -O0 build because the shape they look for could have been written
    that way by hand. This one couldn't: C has no way to say "go to that
    function and don't come back", so a jump standing where a call belongs is
    always something the compiler did.

    A function that tail-calls itself is the case this misses. That one comes
    out as a jump to a label just inside the same function, which is a loop by
    the time it's asm and reads as one — `has_loop_branch` is what finds it.
    """
    line = tail_jump_line(asm)
    if line is None:
        return None

    return Annotation(TAIL_CALL, line, line, TAIL_CALL_DESCRIPTION)


def _call_target(mnemonic: str, operands: list[str]) -> str | None:
    """The function a call names, or None when the line isn't a direct call.

    A call through a pointer — `call *%rax`, or the Intel spelling through
    memory — drops out here, the same way `_jump_target` lets an indirect jump
    fall through. There's no name in the operand to hold against the other
    build, so a callee reached that way can't be followed either side.
    """
    if mnemonic not in CALL_MNEMONICS or len(operands) != 1:
        return None
    target = operands[0]
    return None if _is_indirect(target) else target


def called_functions(asm: str) -> set[str]:
    """Every function the body calls by name, lowercased.

    A set and not a count, because the question underneath is whether a given
    callee is still reached at all, and two calls to the same helper answer
    that once between them. The cost is a function called from two places with
    only one of them inlined: the other call keeps the name in here and the
    half that was copied in goes unreported.
    """
    names: set[str] = set()
    for line in asm.splitlines():
        parsed = _parse_instruction(line)
        if parsed is None:
            continue
        target = _call_target(*parsed)
        if target is not None:
            names.add(target)
    return names


def _outward_jumps(asm: str) -> set[str]:
    """Names the body jumps to that aren't labels it defines itself.

    Which is to say the functions it leaves for without meaning to come back —
    tail calls, by the time they're asm. `tail_jump_line` asks the same thing
    of the last instruction only, since that's where a tail call has to be;
    here every jump counts, because all that matters below is whether the name
    is still reachable somehow.
    """
    local = _local_labels(asm)
    targets: set[str] = set()
    for line in asm.splitlines():
        parsed = _parse_instruction(line)
        if parsed is None:
            continue
        target = _jump_target(*parsed)
        if target is not None and not _is_indirect(target) and target not in local:
            targets.add(target)
    return targets


def inlined_callees(baseline: str, optimized: str) -> set[str]:
    """Functions the baseline calls that the optimized build never reaches.

    Neither by a call nor by a jump. Taking the jumps out is what keeps a tail
    call from landing in here: `call helper` becoming `jmp helper` loses the
    call, but helper still runs, so nothing about it was copied in.
    """
    gone = called_functions(baseline) - called_functions(optimized)
    return gone - _outward_jumps(optimized)


def detect_inlining(baseline: str, optimized: str) -> Annotation | None:
    """Spot a callee whose body was copied into this function.

    A call isn't free. The arguments have to go in the registers the ABI asks
    for, the return address gets pushed, control leaves, and the callee sets up
    a frame of its own before it does any of the work. For a helper that's a
    few instructions long, all of that costs more than the helper does. So the
    compiler writes the helper's instructions straight into the caller and
    deletes the call — and then the two sides are one body, which is what makes
    it worth doing beyond the call itself: constants from here fold into work
    from there, and the optimizations after it see the whole thing at once.

    Two bodies again, for the reason `detect_dead_code_elimination` needs them.
    Inlining leaves no shape behind — the copied instructions are ordinary
    instructions and nothing marks where they came from. All you can see is a
    name that used to be called and isn't. That has to come from the -O0 build.

    A call that turned into a jump is excluded, since the callee is still doing
    its own work and only the return trip went; that one is `detect_tail_call`.
    What isn't excluded is a call the compiler deleted outright, which it may
    do when it can prove the callee has no side effects and nothing reads the
    result. That reads exactly the same from here and gets reported as
    inlining. Same trade as the detectors above: the asm says the callee went,
    not where it went to.

    The annotation covers the whole optimized body rather than a line, because
    the copied instructions are spread through it with nothing separating them
    from the caller's own. Narrowing that down means asking gcc: -fopt-info-inline
    names the callee and the line it was inlined into, so nothing is guessed.
    """
    if not baseline.strip() or not optimized.strip():
        return None

    if not inlined_callees(baseline, optimized):
        return None

    span = _instruction_line_range(optimized)
    if span is None:
        return None

    start, end = span
    return Annotation(INLINING, start, end, INLINING_DESCRIPTION)


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
        parsed = _parse_instruction(line)
        if parsed is None:
            continue
        if _is_conditional_jump(parsed[0]):
            return True
    return False


def _counts_instructions(asm: str, wanted: frozenset[str]) -> int:
    """How many instructions in the body have a stem in `wanted`."""
    total = 0
    for line in asm.splitlines():
        parsed = _parse_instruction(line)
        if parsed is not None and _stem(parsed[0]) in wanted:
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
    and `detect_dead_code_elimination` is the one that reports it.

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

    span = _instruction_line_range(optimized)
    if span is None:
        return None

    start, end = span
    return Annotation(STRENGTH_REDUCTION, start, end, STRENGTH_REDUCTION_DESCRIPTION)


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
    as well if it were asked, so `detect_loop_unrolling` is checked first and
    the ordering in `PAIRED_DETECTORS` is what keeps them apart.
    """
    if not baseline.strip() or not optimized.strip():
        return None

    if not has_conditional_branch(baseline) or has_conditional_branch(optimized):
        return None

    span = _instruction_line_range(optimized)
    if span is None:
        return None

    start, end = span
    return Annotation(BRANCH_ELIMINATION, start, end, BRANCH_ELIMINATION_DESCRIPTION)


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
        parsed = _parse_instruction(line)
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


def _report(console: Console, annotations: list[Annotation], quiet: str) -> None:
    """Print the list of what was found, each with what it means."""
    if not annotations:
        # worth saying out loud — an empty notes column looks the same as a
        # column we forgot to fill in
        console.print("no optimizations detected", style=quiet)
        return

    plural = "" if len(annotations) == 1 else "s"
    console.print(f"\n{len(annotations)} optimization{plural} found:")
    for note in annotations:
        console.print(f"  {note.label()}")
        if note.description:
            console.print(f"    {note.description}", style=quiet)


def _run_explain(console: Console, name: str) -> None:
    """Answer `--explain <opt>`, or list the names when it doesn't match one."""
    known = match_name(name)
    if known is not None:
        console.print(f"{known}: {DESCRIPTIONS[known]}")
        return

    typer.echo(f"error: nothing known as {name!r}", err=True)
    typer.echo(f"known optimizations: {', '.join(DESCRIPTIONS)}", err=True)
    raise typer.Exit(code=1)


def _check_source(path: Path) -> None:
    """Stop on anything we can't hand to a compiler. A plain line beats a traceback."""
    if not path.exists():
        typer.echo(f"error: no such file: {path}", err=True)
        raise typer.Exit(code=1)
    if not path.is_file():
        typer.echo(f"error: not a file: {path}", err=True)
        raise typer.Exit(code=1)


def _baseline_body(cleaned: dict[str, str], func: str | None) -> str:
    """The -O0 body of the wanted function, or a clean error naming what's there.

    The baseline is where a name has to exist, because -O0 does what the source
    said. Anything missing from it was never written rather than optimized out,
    so a miss here is a typo and gets reported as one.
    """
    try:
        return isolate_function(cleaned[BASELINE_LEVEL], func)
    except KeyError:
        names = function_names(cleaned[BASELINE_LEVEL])
        typer.echo(f"error: no function named {func!r}", err=True)
        if names:
            typer.echo(f"available functions: {', '.join(names)}", err=True)
        raise typer.Exit(code=1) from None


def run_annotate(path: Path | None, level: str = "2", func: str | None = None,
                 summary: bool = False, explain_name: str | None = None,
                 no_color: bool = False, width: int | None = None,
                 compiler: str | None = None) -> None:
    """Entry point for `compopt annotate`.

    Compiles the file at -O0 and at the level asked for, runs every detector
    over the pair, and prints the optimized asm with what they found beside it
    and a list underneath. `summary` drops the asm and keeps just the list.

    `explain_name` is the odd one out — it looks up what an optimization means
    by name and prints that, without compiling anything, so it answers "what is
    register coalescing" as well as "what happened to this file". That's why
    the path is optional here.

    Annotating -O0 against itself is allowed and comes back nearly empty, which
    is the honest answer: the paired detectors are comparing a body with
    itself, and the single-body ones are looking at code the optimizer hasn't
    touched.

    The detectors report the shape they see, not what gcc says it did, so some
    of these are guesses — `detect_constant_folding` can't tell a folded
    calculation from a constant that was written that way, and
    `detect_register_coalescing` can't tell a spill that was removed from a
    function that never had one. Their docstrings say which way each one errs.
    """
    console = Console(no_color=no_color, width=width)
    quiet = "" if no_color else "dim"

    if explain_name is not None:
        _run_explain(console, explain_name)
        return

    if path is None:
        typer.echo("error: a source file is required", err=True)
        raise typer.Exit(code=1)

    level = normalize_level(level)
    check_level("--level", level)
    _check_source(path)

    compiler = choose_compiler(compiler)
    levels = list(dict.fromkeys([BASELINE_LEVEL, level]))
    cleaned = {
        name: strip_directives(text)
        for name, text in compile_at_levels(path, compiler, levels).items()
    }

    baseline = _baseline_body(cleaned, func)
    if not baseline.strip():
        console.print(f"no functions to annotate in {path}", style=quiet)
        return

    # empty on the optimized side is a finding, not a mistake: the name was in
    # the baseline, so something the compiler did made it stop being anywhere
    optimized = find_function(cleaned[level], func)
    if not optimized.strip():
        console.print(f"the function is gone at -O{level} (inlined or optimized away)",
                      style=quiet)
        return

    annotations = find_annotations(baseline, optimized)
    if not summary:
        console.print(render_annotated(f"-O{level}", optimized, annotations, color=not no_color))
    _report(console, annotations, quiet)
