"""The show command — eventually compiles a source file and prints the asm.

For now it only checks the file is there. The real compilation comes later.
"""

from pathlib import Path

import typer


def run_show(path: Path) -> None:
    """Entry point for `compopt show`.

    Right now this just makes sure the file exists and echoes the path back.
    """
    if not path.exists():
        # bail out with a non-zero exit instead of a traceback
        typer.echo(f"error: no such file: {path}", err=True)
        raise typer.Exit(code=1)

    if not path.is_file():
        typer.echo(f"error: not a file: {path}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"would show: {path}")
