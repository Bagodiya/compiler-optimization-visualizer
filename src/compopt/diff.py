"""The diff command — compare the assembly of two optimization levels.

This is just the skeleton for now: it checks the source file and reports
which two levels it will compare. The actual line diffing comes later.
"""

from pathlib import Path

import typer


def run_diff(path: Path, from_level: str = "0", to_level: str = "2") -> None:
    """Entry point for `compopt diff`.

    Eventually this compiles the file and shows what changed in the asm
    going from one -O level to another. Right now it only validates the
    input and prints what it's going to do, so the rest of the command
    can be built on top of a command that already exists and is wired in.
    """
    if not path.exists():
        typer.echo(f"error: no such file: {path}", err=True)
        raise typer.Exit(code=1)

    if not path.is_file():
        typer.echo(f"error: not a file: {path}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"would diff -O{from_level} against -O{to_level} for {path}")
