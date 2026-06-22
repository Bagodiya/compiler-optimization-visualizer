"""The show command — compiles a source file and prints its assembly."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from compopt.asm import function_names, isolate_function, strip_directives
from compopt.compilers import compile_at_levels, find_compilers

# the two levels we put side by side for now; the four-column view comes later
COLUMN_LEVELS = ["0", "2"]


def _function_body(asm: str, func: str | None) -> str:
    """Clean one level's assembly and pull out the function we want from it."""
    return isolate_function(strip_directives(asm), func)


def render_columns(columns: list[tuple[str, str]]) -> Table:
    """Lay several assembly bodies out as side-by-side columns.

    Each item in ``columns`` is a (header, body) pair: the header labels the
    column (e.g. ``-O0``) and the body is the already-cleaned assembly for a
    single function. Everything goes into one row of a rich table, which keeps
    the columns aligned to the same top edge so you can scan across the levels.
    Lines are folded rather than cut so nothing silently disappears on a narrow
    terminal.
    """
    table = Table(pad_edge=False)
    for header, _ in columns:
        table.add_column(header, overflow="fold")
    table.add_row(*(body for _, body in columns))
    return table


def run_show(path: Path, func: str | None = None) -> None:
    """Entry point for `compopt show`.

    Compiles the file at every optimization level, then prints the assembly
    for a single function at -O0 and -O2 side by side. Pass ``func`` to pick
    which function; without it we just show the first one in the file.
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

    columns = []
    for level in COLUMN_LEVELS:
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

    Console().print(render_columns(columns))
