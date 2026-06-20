"""The show command — compiles a source file and prints its assembly."""

from pathlib import Path

import typer

from compopt.asm import function_names, isolate_function, strip_directives
from compopt.compilers import compile_at_levels, find_compilers

# the level we print for now; the side-by-side view of all levels comes later
SHOW_LEVEL = "2"


def run_show(path: Path, func: str | None = None) -> None:
    """Entry point for `compopt show`.

    Compiles the file at every optimization level and, for now, prints the
    -O2 assembly for a single function. Pass ``func`` to pick which one;
    without it we just show the first function in the file. Once rendering
    lands this will show the levels side by side.
    """
    if not path.exists():
        # bail out with a non-zero exit instead of a traceback
        typer.echo(f"error: no such file: {path}", err=True)
        raise typer.Exit(code=1)

    if not path.is_file():
        typer.echo(f"error: not a file: {path}", err=True)
        raise typer.Exit(code=1)

    compilers = find_compilers()
    if not compilers:
        typer.echo("error: could not find gcc or clang on PATH", err=True)
        raise typer.Exit(code=1)

    # gcc first if it's around, otherwise whatever we found
    compiler = compilers[0]
    asm = compile_at_levels(path, compiler)

    # clean out the assembler bookkeeping, then pull out the one function
    cleaned = strip_directives(asm[SHOW_LEVEL])
    try:
        body = isolate_function(cleaned, func)
    except KeyError:
        names = function_names(cleaned)
        typer.echo(f"error: no function named {func!r}", err=True)
        if names:
            typer.echo(f"available functions: {', '.join(names)}", err=True)
        raise typer.Exit(code=1) from None

    typer.echo(body)
