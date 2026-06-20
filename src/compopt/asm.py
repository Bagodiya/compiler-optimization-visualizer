"""Cleaning up the bookkeeping the assembler emits around the real code."""

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


def _label_name(line: str) -> str | None:
    """Return the function name a line opens, or None if it isn't one.

    A function label sits at column 0 (no indentation) and ends with a colon,
    e.g. ``add:``. The compiler also drops in its own local labels like
    ``.LFB0:`` while it works — those start with a dot, so we skip them. On
    macOS clang prefixes names with an underscore (``_add:``), tacks a comment
    onto the same line (``_add:    ## @add``), and emits comment-only lines
    like ``## %bb.0:``. So we strip any trailing comment first and bail on
    lines that are pure comments.
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
    if not name or name.startswith("."):
        return None
    return name


def _is_function_label(line: str) -> bool:
    """True if a line opens a real function, e.g. ``add:``."""
    return _label_name(line) is not None


def _matches(symbol: str, wanted: str) -> bool:
    """Whether ``symbol`` is the function the user asked for by name.

    Handles the macOS underscore prefix so ``--func add`` finds ``_add``.
    """
    return symbol == wanted or symbol.lstrip("_") == wanted


def function_names(asm: str) -> list[str]:
    """Return the names of the top-level functions, in the order they show up."""
    names = []
    for line in asm.splitlines():
        name = _label_name(line)
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
    lines = asm.splitlines()
    starts = [i for i, line in enumerate(lines) if _is_function_label(line)]
    if not starts:
        return ""

    if name is None:
        begin = starts[0]
    else:
        begin = next(
            (i for i in starts if _matches(_label_name(lines[i]), name)), None
        )
        if begin is None:
            raise KeyError(name)

    # stop at the next function, or run to the end if this is the last one
    end = next((i for i in starts if i > begin), len(lines))
    return "\n".join(lines[begin:end]).rstrip()
