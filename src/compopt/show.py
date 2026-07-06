"""The show command — compiles a source file and prints its assembly."""

from pathlib import Path

import typer
from rich.console import Console

from compopt.asm import function_names, isolate_function, strip_directives
from compopt.compilers import compile_at_levels, find_compilers
from compopt.render import levels_for_width, render_columns


def _function_body(asm: str, func: str | None) -> str:
    """Clean one level's assembly and pull out the function we want from it."""
    return isolate_function(strip_directives(asm), func)


def run_show(
    path: Path,
    func: str | None = None,
    no_color: bool = False,
    width: int | None = None,
) -> None:
    """Entry point for `compopt show`.

    Compiles the file at every optimization level, then prints the assembly
    for a single function side by side. On a wide terminal that's all four
    levels (-O0..-O3); on a narrower one we show -O0 vs -O2. Pass ``func`` to
    pick which function; without it we just show the first one in the file.
    Set ``no_color`` to get plain output with the highlighting turned off.
    Pass ``width`` to force a column count instead of measuring the terminal,
    which is handy for a fixed layout or when the output is being piped.
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

    # width=None lets rich measure the terminal; a number pins it instead
    console = Console(no_color=no_color, width=width)
    levels = levels_for_width(console.width)

    columns = []
    for level in levels:
        try:
            body = _function_body(asm[level], func)
        except KeyError:
            # a missing function fails the same way at every level, so the
            # first miss is enough to report what is actually available
            names = function_names(strip_directives(asm[level]))
            typer.echo(f"error: no function named {func!r}", err=True)
            if names:
                typer.echo(f"available functions: {', '.join(names)}", err=True)
            raise typer.Exit(code=1) from None
        columns.append((f"-O{level}", body))

    console.print(render_columns(columns, color=not no_color))
