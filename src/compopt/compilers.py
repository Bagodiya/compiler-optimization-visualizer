"""Figuring out which compilers we can actually use on this machine."""

import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# the ones we know how to drive for now
KNOWN_COMPILERS = ["gcc", "clang"]

# the optimization levels we compare by default
DEFAULT_LEVELS = ["0", "1", "2", "3"]


class CompileError(Exception):
    """Raised when the compiler refuses to build the source.

    Carries whatever the compiler printed to stderr so the caller can
    show the user something useful instead of a stack trace.
    """

    def __init__(self, compiler: str, message: str) -> None:
        self.compiler = compiler
        self.message = message.strip()
        super().__init__(self.message)


def find_compilers() -> list[str]:
    """Return the compilers from KNOWN_COMPILERS that are on PATH.

    Uses shutil.which so we only report compilers we can really run.
    Order follows KNOWN_COMPILERS, gcc first.
    """
    found = []
    for name in KNOWN_COMPILERS:
        if shutil.which(name) is not None:
            found.append(name)
    return found


def compile_to_asm(source: Path, level: str, compiler: str) -> str:
    """Compile one source file at a single -O level and give back the asm.

    `level` is just the digit, so "2" turns into -O2. We ask the compiler
    for assembly (-S), drop it in a throwaway temp dir and read it back.
    The temp dir is removed once we have the text so nothing piles up.
    """
    with tempfile.TemporaryDirectory(prefix="compopt-") as workdir:
        out = Path(workdir) / "out.s"

        cmd = [compiler, "-S", f"-O{level}", str(source), "-o", str(out)]
        # don't use check=True here: we want to grab stderr and wrap it
        # in our own error rather than let CalledProcessError escape.
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = result.stderr or result.stdout or "compilation failed"
            raise CompileError(compiler, detail)

        return out.read_text()


def compile_at_levels(
    source: Path, compiler: str, levels: list[str] | None = None
) -> dict[str, str]:
    """Compile the same source at several -O levels and return them keyed by level.

    Defaults to O0/O1/O2/O3. Each level is an independent compiler run, and
    since those are mostly waiting on the compiler process we just fan them
    out across a thread pool instead of doing them one after another.

    If any level fails to compile the CompileError propagates — there's no
    point showing a half-finished comparison.
    """
    if levels is None:
        levels = DEFAULT_LEVELS

    with ThreadPoolExecutor(max_workers=len(levels)) as pool:
        # keep the future->level mapping so we can label results correctly
        futures = {
            pool.submit(compile_to_asm, source, level, compiler): level
            for level in levels
        }
        return {level: fut.result() for fut, level in futures.items()}
