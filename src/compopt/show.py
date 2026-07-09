"""The show command — compiles a source file and prints its assembly."""

import os
from pathlib import Path

import typer
from rich.console import Console

from compopt.asm import function_names, isolate_function, strip_directives
from compopt.compilers import compile_at_levels, find_compilers
from compopt.render import levels_for_width, render_columns


def _function_body(asm: str, func: str | None) -> str:
    """Clean one level's assembly and pull out the function we want from it."""
    return isolate_function(strip_directives(asm), func)


def _pick_compiler(requested: str | None, available: list[str]) -> str:
    """Work out which compiler to actually run.

    An explicit --compiler wins but has to really be installed, otherwise
    we stop. With no flag we look at $CC the same way make and configure do,
    so `CC=clang compopt show foo.c` just works. $CC can be a bare name or a
    full path like /usr/bin/clang, so we compare on the file name. Anything
    we can't drive (say CC=cc) is ignored with a warning and we fall back to
    gcc-first.
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


def run_show(
    path: Path,
    func: str | None = None,
    no_color: bool = False,
    width: int | None = None,
    compiler: str | None = None,
) -> None:
    """Entry point for `compopt show`.

    Compiles the file at every optimization level, then prints the assembly
    for a single function side by side. On a wide terminal that's all four
    levels (-O0..-O3); on a narrower one we show -O0 vs -O2. Pass ``func`` to
    pick which function; without it we just show the first one in the file.
    Set ``no_color`` to get plain output with the highlighting turned off.
    Pass ``width`` to force a column count instead of measuring the terminal,
    which is handy for a fixed layout or when the output is being piped.
    Pass ``compiler`` to force gcc or clang. With nothing forced we honour
    the $CC environment variable, and if that isn't set either we fall back
    to whichever we find first (gcc when it's around).
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

    compiler = _pick_compiler(compiler, compilers)

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
