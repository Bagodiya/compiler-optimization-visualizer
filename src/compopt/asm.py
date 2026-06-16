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
