"""Cleaning up the bookkeeping the assembler emits around the real code."""

import re

# Pure housekeeping directives the compiler emits that tell us nothing about
# how the code was optimized. Dropping these makes the output much easier to
# read (and, later, to diff). Anything that starts with one of these prefixes
# gets thrown away.
NOISE_DIRECTIVES = (
    ".file",
    ".ident",
    ".section",
    ".text",
    ".data",
    ".bss",
    ".globl",
    ".global",
    ".local",
    ".type",
    ".size",
    ".align",
    ".p2align",
    ".cfi_",  # call-frame info — there are a lot of these
    ".addrsig",
)


# On ELF a local label starts with a dot (`.L2:`) and the check in
# `_label_name` catches it. Mach-O doesn't work that way: its local labels are a
# bare `L` at column 0, which is exactly where a function label goes. clang
# writes `LBB0_1:` for a basic block, GNU gcc writes `LFB0:` and `LFE0:` around
# every function, and both also emit things like `EH_frame1:` on the way past.
# Counted as functions, those do real damage — `isolate_function` stops at the
# next function label, so a function gets cut off at its own first local label
# and everything below it quietly disappears.
#
# The rule that sorts it out isn't a list of prefixes to skip, it's that Mach-O
# gives every C symbol a leading underscore. So in Mach-O output a function
# label is one that starts with `_`, and everything else at column 0 is the
# assembler talking to itself.
_MACHO_SYMBOL = re.compile(r"^_\w+:", re.M)
_ELF_LOCAL_LABEL = re.compile(r"^\.L\w*:", re.M)


def _is_macho(asm: str) -> bool:
    """Whether this assembly is Mach-O, which decides what a label means.

    Underscored symbols say Mach-O and dotted local labels say ELF, and asking
    for both is what keeps the guess honest — a Linux function that happens to
    be called `_start` would otherwise be enough on its own to switch the rule
    over and hide every function that isn't underscored.
    """
    return bool(_MACHO_SYMBOL.search(asm)) and not _ELF_LOCAL_LABEL.search(asm)


def _is_noise(line: str) -> bool:
    """True if a (stripped) line is just a noise directive we want gone."""
    if not line.startswith("."):
        # instruction, label, or blank line — always keep it
        return False
    return line.startswith(NOISE_DIRECTIVES)


def strip_directives(asm: str) -> str:
    """Remove noisy assembler directives, keeping instructions and labels.

    Walks the assembly line by line and drops anything that is just one of
    the directives in NOISE_DIRECTIVES. Indentation on the kept lines is left
    alone so the output still lines up the way the compiler wrote it. Labels
    (.L...) are kept since those aren't noise.
    """
    kept = [line for line in asm.splitlines() if not _is_noise(line.strip())]
    return "\n".join(kept)


def _label_name(line: str, macho: bool = False) -> str | None:
    """Return the function name a line opens, or None if it isn't one.

    A function label sits at column 0 (no indentation) and ends with a colon,
    e.g. ``add:``. The compiler also drops in its own local labels like
    ``.LFB0:`` while it works — those start with a dot, so we skip them. On
    macOS clang prefixes names with an underscore (``_add:``), tacks a comment
    onto the same line (``_add:    ## @add``), and emits comment-only lines
    like ``## %bb.0:``. So we strip any trailing comment first and bail on
    lines that are pure comments.

    With ``macho`` set the test flips from "doesn't start with a dot" to "does
    start with an underscore", which is the only way to tell a function from
    the assembler's own labels in that output — see `_is_macho` above.
    """
    if not line or line[0].isspace():
        # indented => it's an instruction, not a label
        return None
    if line.lstrip().startswith("#"):
        # whole line is a comment (clang's ## %bb.0:, gcc's # comments)
        return None
    name = line.split("#", 1)[0].rstrip()
    if not name.endswith(":"):
        return None
    name = name[:-1]
    if not name:
        return None
    if macho:
        return name if name.startswith("_") else None
    return None if name.startswith(".") else name


def _is_function_label(line: str, macho: bool = False) -> bool:
    """True if a line opens a real function, e.g. ``add:``."""
    return _label_name(line, macho) is not None


def _matches(symbol: str, wanted: str) -> bool:
    """Whether ``symbol`` is the function the user asked for by name.

    Handles the macOS underscore prefix so ``--func add`` finds ``_add``.
    """
    return symbol == wanted or symbol.lstrip("_") == wanted


def function_names(asm: str) -> list[str]:
    """Return the names of the top-level functions, in the order they show up."""
    macho = _is_macho(asm)
    names = []
    for line in asm.splitlines():
        name = _label_name(line, macho)
        if name is not None:
            names.append(name)
    return names


def isolate_function(asm: str, name: str | None = None) -> str:
    """Pull out the lines belonging to a single function.

    Grabs everything from the function's label down to (but not including)
    the next function label, so the local ``.L`` labels in between come along
    for the ride. With no name we just take the first function we find, which
    is usually the one you care about in these little example files.

    Raises KeyError if a name is given but no such function exists.
    """
    macho = _is_macho(asm)
    lines = asm.splitlines()
    starts = [i for i, line in enumerate(lines) if _is_function_label(line, macho)]
    if not starts:
        return ""

    if name is None:
        begin = starts[0]
    else:
        begin = next(
            (i for i in starts if _matches(_label_name(lines[i], macho), name)), None
        )
        if begin is None:
            raise KeyError(name)

    # stop at the next function, or run to the end if this is the last one
    end = next((i for i in starts if i > begin), len(lines))
    return "\n".join(lines[begin:end]).rstrip()


def find_function(asm: str, name: str | None = None) -> str:
    """Same as `isolate_function`, but a missing function is just "".

    `show` wants the KeyError: if you ask for a function that isn't there,
    that's a typo and it should say so. `diff` is the opposite — the function
    disappearing at the higher level is the interesting result, not an error,
    since that's what inlining looks like from the outside. So it gets this
    version and decides for itself what to print.
    """
    try:
        return isolate_function(asm, name)
    except KeyError:
        return ""
