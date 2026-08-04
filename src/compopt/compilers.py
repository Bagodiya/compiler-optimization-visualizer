"""Figuring out which compilers we can actually use on this machine."""

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

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


def normalize_level(level: str) -> str:
    """Take a level however the user spelled it and give back the bare digit.

    Everything inside passes levels around as `"0"`..`"3"` and only sticks the
    `-O` on at display time, but nobody thinks of them that way — the flag you
    have in your head is `-O2`, so that's what gets typed. Accepting `2`, `O2`
    and `-O2` costs three calls and saves the reader from a rejection that
    looks like the tool doesn't know its own levels.

    A spelling we don't recognise is passed through untouched rather than
    patched up, so `--to fast` still reaches `check_level` and gets reported
    against the list of levels instead of quietly becoming something else.
    """
    return level.removeprefix("-").removeprefix("O").removeprefix("o") or level


def check_level(flag: str, level: str) -> None:
    """Stop early if a level isn't one we know how to compile.

    Only the digits in DEFAULT_LEVELS are valid, so `--from 9` is caught here
    instead of turning into a `-O9` the compiler would reject. `flag` names the
    option it came from, since the message is no use if you passed two of them
    and can't tell which one it's complaining about.
    """
    if level not in DEFAULT_LEVELS:
        typer.echo(f"error: {flag} must be one of: {', '.join(DEFAULT_LEVELS)}", err=True)
        raise typer.Exit(code=1)


def pick_compiler(requested: str | None, available: list[str]) -> str:
    """Work out which compiler to actually run.

    An explicit --compiler wins but has to really be installed, otherwise
    we stop. With no flag we look at $CC the same way make and configure do,
    so `CC=clang compopt show foo.c` just works. $CC can be a bare name or a
    full path like /usr/bin/clang, so we compare on the file name. Anything
    we can't drive (say CC=cc) is ignored with a warning and we fall back to
    gcc-first.

    Takes the available list rather than calling `find_compilers` itself, so
    the choosing can be tested without a toolchain installed.
    """
    if requested is not None:
        if requested not in available:
            typer.echo(f"error: {requested} is not available on PATH", err=True)
            typer.echo(f"available: {', '.join(available)}", err=True)
            raise typer.Exit(code=1)
        return requested

    env_cc = os.environ.get("CC")
    if env_cc:
        name = Path(env_cc).name
        if name in available:
            return name
        typer.echo(
            f"warning: ignoring $CC={env_cc}, not one of: {', '.join(available)}",
            err=True,
        )

    # gcc first if it's around, otherwise whatever we found
    return available[0]


def choose_compiler(requested: str | None) -> str:
    """Find what's installed and settle on one, or stop if there's nothing.

    Every command opens the same way — look at the machine, then honour
    whatever the user asked for — so the two steps live together here rather
    than being spelled out three times over.
    """
    available = find_compilers()
    if not available:
        typer.echo("error: could not find gcc or clang on PATH", err=True)
        raise typer.Exit(code=1)
    return pick_compiler(requested, available)


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
