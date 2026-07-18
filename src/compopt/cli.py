"""Command-line interface for compopt."""

from pathlib import Path
from typing import Annotated

import typer

from compopt import __version__
from compopt.compilers import CompileError
from compopt.diff import run_diff
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
    func: Annotated[
        str | None,
        typer.Option("--func", "-f", help="Which function to display."),
    ] = None,
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="Disable color in the output."),
    ] = False,
    width: Annotated[
        int | None,
        typer.Option("--width", help="Force the output width instead of measuring the terminal."),
    ] = None,
    compiler: Annotated[
        str | None,
        typer.Option("--compiler", "-c", help="Which compiler to use: gcc or clang."),
    ] = None,
) -> None:
    """Show the optimized output for a source file."""
    try:
        run_show(path, func, no_color, width, compiler)
    except CompileError as err:
        # the compiler already told us what's wrong, just pass it along
        typer.echo(f"error: {err.compiler} could not compile {path}", err=True)
        typer.echo(err.message, err=True)
        raise typer.Exit(code=1) from err


@app.command()
def diff(
    path: Annotated[Path, typer.Argument(help="C source file to inspect.")],
    # "from" is a keyword so the flag name has to be spelled out here
    from_level: Annotated[
        str,
        typer.Option("--from", help="Level to compare from, as a bare digit."),
    ] = "0",
    to_level: Annotated[
        str,
        typer.Option("--to", help="Level to compare to, as a bare digit."),
    ] = "2",
    context: Annotated[
        int,
        typer.Option("--context", "-C", help="Unchanged lines to keep around each change."),
    ] = 3,
    unified: Annotated[
        bool,
        typer.Option("--unified", "-u", help="Emit a standard unified diff (diff -u format)."),
    ] = False,
) -> None:
    """Show what changed in the asm between two optimization levels."""
    run_diff(path, from_level, to_level, context, unified)


if __name__ == "__main__":
    app()
