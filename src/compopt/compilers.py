"""Figuring out which compilers we can actually use on this machine."""

import shutil
import subprocess
import tempfile
from pathlib import Path

# the ones we know how to drive for now
KNOWN_COMPILERS = ["gcc", "clang"]


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
    for assembly (-S), drop it in a temp dir and read it back as text.
    """
    workdir = tempfile.mkdtemp(prefix="compopt-")
    out = Path(workdir) / "out.s"

    cmd = [compiler, "-S", f"-O{level}", str(source), "-o", str(out)]
    subprocess.run(cmd, check=True)

    return out.read_text()
