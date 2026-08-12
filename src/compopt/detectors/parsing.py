"""Reading instructions, and the register names more than one detector needs.

Nothing in here decides anything about optimization. It's the layer underneath
that: turn a line of asm into something comparable, and give the two syntaxes
one spelling so the detectors don't each have to know about both.

Anything used by a single detector stays in that detector's own file. What
lands here is what two or more of them ask for.
"""

# The same prologue gets written two different ways depending on the syntax:
#
#     AT&T     pushq %rbp   then   movq %rsp, %rbp
#     Intel    push  rbp    then   mov  rbp, rsp
#
# rather than handle every spelling separately, the parsing below lowercases
# everything and drops the AT&T '%' so both flavours come out looking the same.

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

# instructions that mean the machine is still working something out at run
# time. only the stems are listed: the sized spellings (addl, subq, imull...)
# come off the same stem, and there are far too many combinations to write out
# the way the prologue mnemonics in `frame.py` are.
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


def parse_instruction(line: str) -> tuple[str, list[str]] | None:
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


def label_name(line: str) -> str | None:
    """The label a line defines, lowercased, or None when it defines none.

    `parse_instruction` throws these away with the directives, since a label
    isn't code, but a jump target is a label and the detectors that follow
    jumps have to look at them. Lowercased for the same reason the operands
    are: it's the only way `.L2` and the `.l2` an operand comes back as
    compare equal.
    """
    text = line.split("#", 1)[0].strip()
    if not text.endswith(":"):
        return None
    name = text.removesuffix(":").strip().lower()
    return name or None


def jump_target(mnemonic: str, operands: list[str]) -> str | None:
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


def instruction_line_range(asm: str) -> tuple[int, int] | None:
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
        if parse_instruction(line) is None:
            continue
        if first is None:
            first = number
        last = number
    if first is None or last is None:
        return None
    return first, last


def first_instruction_line(asm: str) -> int | None:
    """Which line the function's first real instruction sits on, or None."""
    span = instruction_line_range(asm)
    return None if span is None else span[0]


def stem(mnemonic: str) -> str:
    """A mnemonic with its size suffix off: `addl` and `addq` both give `add`.

    Only suffixes that leave something recognisable behind get dropped, so
    `xorps` stays whole instead of turning into an integer xor.
    """
    if mnemonic not in ARITHMETIC_STEMS and mnemonic.endswith(SIZE_SUFFIXES):
        trimmed = mnemonic[:-1]
        if trimmed in ARITHMETIC_STEMS:
            return trimmed
    return mnemonic
