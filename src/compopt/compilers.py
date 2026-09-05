"""Figuring out which compilers we can actually use on this machine."""

import os
import platform
import re
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


# gcc-15, clang-17, gcc-15.1 — a version stuck on the end and nothing else.
# gcc-ar, clang++ and clang-format all start the same way and none of them
# compiles anything, so matching the whole name is what keeps them out.
SUFFIXED = re.compile(r"^(gcc|clang)-[0-9][0-9.]*$")


def suffixed_compilers() -> list[str]:
    """Find versioned compilers on PATH that find_compilers walks straight past.

    Homebrew installs its gcc as `gcc-15` and leaves `gcc` alone, so a machine
    can have a perfectly good compiler on PATH and still get told there isn't
    one. We only look for the bare names, which is a real limitation and not
    one the user can be expected to guess, so the error says the names it
    found instead of pretending the machine is empty.

    Only walked when we're already about to fail, so the cost doesn't matter.
    """
    found = set()
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        try:
            names = os.listdir(entry)
        except OSError:
            # unreadable or missing PATH entries are normal, just skip them
            continue
        found.update(name for name in names if SUFFIXED.match(name))
    return sorted(found)


def install_hint() -> str:
    """The line telling the user how to get a compiler on this kind of machine.

    Three cases is as far as this is worth taking. macOS has one answer that
    is nearly always right, Linux has one per distro so the message names two
    and trusts the reader, and anything else gets told what's needed without
    guessing at how.
    """
    match platform.system():
        case "Darwin":
            return "install the command line tools with: xcode-select --install"
        case "Linux":
            return "install one with your package manager, e.g. apt install gcc"
        case _:
            return "install gcc or clang and make sure it is on PATH"


def no_compiler_lines() -> list[str]:
    """Build the whole "nothing to compile with" message, one line per line.

    Says what was looked for rather than just what wasn't found, because the
    old wording ("could not find gcc or clang on PATH") reads like the machine
    has no compiler when usually it means the one it has is named something
    else. Anything suffixed gets listed, so `gcc-15` sitting there unused is
    on screen instead of being something you have to already know about.

    Returns lines so the tests can read them without capturing stderr.
    """
    lines = [
        f"error: no compiler found — looked for {' and '.join(KNOWN_COMPILERS)} on PATH"
    ]
    nearby = suffixed_compilers()
    if nearby:
        lines.append(f"  PATH does have {', '.join(nearby)}, but only the bare")
        lines.append("  names are used, so rename or symlink one of those into place")
    else:
        lines.append(f"  {install_hint()}")
    return lines


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


def report_bad_request(requested: str, available: list[str]) -> None:
    """Explain a --compiler we can't honour and stop. Never returns.

    Two different things go wrong here and the fix isn't the same for either.
    A name we've never heard of stays broken however much you install, so it
    gets told which names exist at all; a known compiler that just isn't here
    gets the same install advice as having none. Lumping them together as
    "not available on PATH" told the `--compiler tcc` case to go install tcc.
    """
    if requested not in KNOWN_COMPILERS:
        typer.echo(f"error: don't know how to drive {requested}", err=True)
        typer.echo(f"  --compiler takes one of: {', '.join(KNOWN_COMPILERS)}", err=True)
    else:
        typer.echo(f"error: {requested} is not on PATH", err=True)
        if available:
            typer.echo(f"  found instead: {', '.join(available)}", err=True)
        typer.echo(f"  {install_hint()}", err=True)
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
            report_bad_request(requested, available)
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
        for line in no_compiler_lines():
            typer.echo(line, err=True)
        raise typer.Exit(code=1)
    return pick_compiler(requested, available)


def compile_to_asm(source: Path, level: str, compiler: str, debug: bool = False) -> str:
    """Compile one source file at a single -O level and give back the asm.

    `level` is just the digit, so "2" turns into -O2. We ask the compiler
    for assembly (-S), drop it in a throwaway temp dir and read it back.
    The temp dir is removed once we have the text so nothing piles up.

    With `debug` set we add -g, which sprinkles .file and .loc directives
    through the output saying which line of C each run of instructions came
    from. `crossref` needs those to put the report next to the right
    instructions. It's off by default because it makes the asm a good deal
    noisier to read, and -g doesn't change the code that gets generated, so
    the two spellings are the same program either way.
    """
    with tempfile.TemporaryDirectory(prefix="compopt-") as workdir:
        out = Path(workdir) / "out.s"

        cmd = [compiler, "-S", f"-O{level}", str(source), "-o", str(out)]
        if debug:
            cmd.insert(1, "-g")
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


def compiler_version(name: str) -> str:
    """Ask a compiler what it is and hand back the one line worth printing.

    `--version` is the spelling both gcc and clang answer to, and both put the
    part you want on the first line — the rest is the target triple and the
    install directory. Which is the whole point of printing it: the `gcc` on
    this Mac says "Apple clang" when you ask, and nothing else we show tells
    you that.

    A compiler that fails or hangs here comes back as "unknown" instead of
    raising. `--info` is what you run when something is already wrong, so it
    has to survive one broken entry on PATH.
    """
    try:
        result = subprocess.run(
            [name, "--version"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"

    if result.returncode != 0:
        return "unknown"

    lines = result.stdout.strip().splitlines()
    return lines[0].strip() if lines else "unknown"
