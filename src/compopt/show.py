"""The show command — compiles a source file and prints its assembly."""

import re
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from compopt.asm import function_names, isolate_function, strip_directives
from compopt.compilers import compile_at_levels, find_compilers

# every level we can show; the full four-column view needs a wide terminal
ALL_LEVELS = ["0", "1", "2", "3"]

# what we fall back to when there isn't room for all four
NARROW_LEVELS = ["0", "2"]

# rough number of characters one asm column needs before it starts to look
# cramped. picked by eye from typical instruction lines like "movq -8(%rbp), %rax"
MIN_COLUMN_WIDTH = 26


def levels_for_width(width: int) -> list[str]:
    """Decide which -O levels to show given how wide the terminal is.

    Four columns side by side only really work on a wide screen. If we tried
    to cram them into a narrow terminal every line would get folded and the
    whole thing turns into soup, so below the threshold we drop back to just
    -O0 vs -O2.
    """
    if width >= MIN_COLUMN_WIDTH * len(ALL_LEVELS):
        return ALL_LEVELS
    return NARROW_LEVELS


def _function_body(asm: str, func: str | None) -> str:
    """Clean one level's assembly and pull out the function we want from it."""
    return isolate_function(strip_directives(asm), func)


def _highlight_line(line: str) -> Text:
    """Colorize a single line of AT&T assembly.

    The colors are just there to make the output easier to skim:
    labels stand out, the mnemonic is one color and the operands another.
    Anything we don't recognise is left in the default color.
    """
    text = Text(line)
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        # blank line or a comment the compiler left behind — leave it alone
        return text

    # clang sometimes tacks a comment onto the label line ("add:  ## @add"),
    # so look at the code before any '#' when deciding what this line is.
    code = stripped.split("#", 1)[0].rstrip()
    if code.endswith(":") and ":" not in code[:-1]:
        # a label block opener like "add:" or ".L2:" — color just the label,
        # not whatever comment trails behind it
        end = line.index(":") + 1
        text.stylize("bold cyan", 0, end)
        return text

    # an instruction line: first word after the indent is the mnemonic
    head = re.match(r"(\s*)(\w+)", line)
    if head:
        text.stylize("green", head.start(2), head.end(2))
    text.highlight_regex(r"%\w+", "magenta")      # registers, e.g. %rax
    text.highlight_regex(r"\$-?\w+", "yellow")    # immediates, e.g. $5
    return text


def highlight_asm(body: str) -> Text:
    """Turn an assembly body into colored text, line by line."""
    out = Text()
    for i, line in enumerate(body.splitlines()):
        if i:
            out.append("\n")
        out.append_text(_highlight_line(line))
    return out


def line_number_gutter(count: int) -> Text:
    """Build the line numbers that run down the left margin.

    The numbers are right-aligned so the digits stay lined up once the count
    rolls past 9, and dimmed because they're just a reference for pointing at a
    row, not part of the assembly itself.
    """
    pad = len(str(count))
    out = Text()
    for n in range(1, count + 1):
        if n > 1:
            out.append("\n")
        out.append(str(n).rjust(pad), style="dim")
    return out


def render_columns(columns: list[tuple[str, str]]) -> Table:
    """Lay several assembly bodies out as side-by-side columns.

    Each item in ``columns`` is a (header, body) pair: the header labels the
    column (e.g. ``-O0``) and the body is the already-cleaned assembly for a
    single function. Everything goes into one row of a rich table, which keeps
    the columns aligned to the same top edge so you can scan across the levels.
    A narrow gutter of line numbers sits on the far left so the rows are easy to
    refer back to. Lines are folded rather than cut so nothing silently
    disappears on a narrow terminal.
    """
    table = Table(pad_edge=False)
    # number up to the longest body; shorter columns just run out of rows
    rows = max((len(body.splitlines()) for _, body in columns), default=0)
    table.add_column("", justify="right")
    for header, _ in columns:
        table.add_column(header, overflow="fold")
    table.add_row(
        line_number_gutter(rows),
        *(highlight_asm(body) for _, body in columns),
    )
    return table


def run_show(path: Path, func: str | None = None) -> None:
    """Entry point for `compopt show`.

    Compiles the file at every optimization level, then prints the assembly
    for a single function side by side. On a wide terminal that's all four
    levels (-O0..-O3); on a narrower one we show -O0 vs -O2. Pass ``func`` to
    pick which function; without it we just show the first one in the file.
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

    console = Console()
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

    console.print(render_columns(columns))
