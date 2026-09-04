"""What `--info` prints: the version, and what we found to compile with."""

import platform
import shutil

import typer

from compopt import __version__
from compopt.compilers import compiler_version, find_compilers


def compiler_lines() -> list[str]:
    """One line per compiler on PATH: the name we call it, where it is, what it says.

    The path matters as much as the version does. `gcc` is whatever comes
    first on PATH, and on this machine that's Apple clang in /usr/bin while
    the real gcc sits in homebrew under a suffixed name we never look for, so
    the two columns together are what explains a report that came back the
    wrong way round.

    Returns lines rather than printing them so the test can read them back.
    """
    found = find_compilers()
    if not found:
        return ["  none found on PATH"]

    lines = []
    for name in found:
        path = shutil.which(name) or "?"
        lines.append(f"  {name:<6} {path:<20} {compiler_version(name)}")
    return lines


def run_info() -> None:
    """Print everything worth knowing before filing a bug about this tool.

    Runs each compiler once for its version banner, so it's slower than
    `--version` — that's fine, nothing else waits on it.
    """
    typer.echo(f"compopt {__version__}")
    typer.echo(f"python {platform.python_version()} on {platform.system().lower()}")
    typer.echo("compilers:")
    for line in compiler_lines():
        typer.echo(line)
