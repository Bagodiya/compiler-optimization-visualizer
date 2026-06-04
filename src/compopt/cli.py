"""Command-line interface for compopt."""

from pathlib import Path
from typing import Annotated

import typer

from compopt import __version__
from compopt.show import run_show

app = typer.Typer(
    name="compopt",
    help=(
        "Inspect and compare compiler optimization output.\n\n"
        "Run gcc or clang at different -O levels and see what changed "
        "in the generated assembly."
    ),
    no_args_is_help=True,
    add_completion=False,
)


def show_version(value: bool) -> None:
    # eager callback so --version works before any command runs
    if value:
        typer.echo(f"compopt {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=show_version,
        is_eager=True,
    ),
) -> None:
    """Compare what the compiler does at each optimization level."""


@app.command()
def version() -> None:
    """Print the compopt version."""
    typer.echo(f"compopt {__version__}")


@app.command()
def show(
    path: Annotated[Path, typer.Argument(help="C source file to inspect.")],
) -> None:
    """Show the optimized output for a source file."""
    run_show(path)


if __name__ == "__main__":
    app()
