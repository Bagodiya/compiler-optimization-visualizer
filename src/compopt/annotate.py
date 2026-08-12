"""The annotate command: compile a file, run the detectors, print what they say.

The detecting itself lives in `compopt.detectors`, one optimization per module.
What's left here is the command around them — working out which two levels to
compile, pulling the wanted function out of each, and printing the result.
"""

from pathlib import Path

import typer
from rich.console import Console

from compopt.annotation import Annotation
from compopt.asm import find_function, function_names, isolate_function, strip_directives
from compopt.compilers import (
    check_level,
    choose_compiler,
    compile_at_levels,
    normalize_level,
)
from compopt.detectors import DESCRIPTIONS, find_annotations, match_name
from compopt.render import render_annotated

# the level everything is compared against. the detectors that need two bodies
# need one where the compiler hasn't done anything yet, and that's -O0.
BASELINE_LEVEL = "0"


def _report(console: Console, annotations: list[Annotation], quiet: str) -> None:
    """Print the list of what was found, each with what it means."""
    if not annotations:
        # worth saying out loud — an empty notes column looks the same as a
        # column we forgot to fill in
        console.print("no optimizations detected", style=quiet)
        return

    plural = "" if len(annotations) == 1 else "s"
    console.print(f"\n{len(annotations)} optimization{plural} found:")
    for note in annotations:
        console.print(f"  {note.label()}")
        if note.description:
            console.print(f"    {note.description}", style=quiet)


def _run_explain(console: Console, name: str) -> None:
    """Answer `--explain <opt>`, or list the names when it doesn't match one."""
    known = match_name(name)
    if known is not None:
        console.print(f"{known}: {DESCRIPTIONS[known]}")
        return

    typer.echo(f"error: nothing known as {name!r}", err=True)
    typer.echo(f"known optimizations: {', '.join(DESCRIPTIONS)}", err=True)
    raise typer.Exit(code=1)


def _check_source(path: Path) -> None:
    """Stop on anything we can't hand to a compiler. A plain line beats a traceback."""
    if not path.exists():
        typer.echo(f"error: no such file: {path}", err=True)
        raise typer.Exit(code=1)
    if not path.is_file():
        typer.echo(f"error: not a file: {path}", err=True)
        raise typer.Exit(code=1)


def _baseline_body(cleaned: dict[str, str], func: str | None) -> str:
    """The -O0 body of the wanted function, or a clean error naming what's there.

    The baseline is where a name has to exist, because -O0 does what the source
    said. Anything missing from it was never written rather than optimized out,
    so a miss here is a typo and gets reported as one.
    """
    try:
        return isolate_function(cleaned[BASELINE_LEVEL], func)
    except KeyError:
        names = function_names(cleaned[BASELINE_LEVEL])
        typer.echo(f"error: no function named {func!r}", err=True)
        if names:
            typer.echo(f"available functions: {', '.join(names)}", err=True)
        raise typer.Exit(code=1) from None


def run_annotate(path: Path | None, level: str = "2", func: str | None = None,
                 summary: bool = False, explain_name: str | None = None,
                 no_color: bool = False, width: int | None = None,
                 compiler: str | None = None) -> None:
    """Entry point for `compopt annotate`.

    Compiles the file at -O0 and at the level asked for, runs every detector
    over the pair, and prints the optimized asm with what they found beside it
    and a list underneath. `summary` drops the asm and keeps just the list.

    `explain_name` is the odd one out — it looks up what an optimization means
    by name and prints that, without compiling anything, so it answers "what is
    register coalescing" as well as "what happened to this file". That's why
    the path is optional here.

    Annotating -O0 against itself is allowed and comes back nearly empty, which
    is the honest answer: the paired detectors are comparing a body with
    itself, and the single-body ones are looking at code the optimizer hasn't
    touched.

    The detectors report the shape they see, not what gcc says it did, so some
    of these are guesses — `folding.detect_constant_folding` can't tell a
    folded calculation from a constant that was written that way, and
    `registers.detect_register_coalescing` can't tell a spill that was removed
    from a function that never had one. Their docstrings say which way each one
    errs.
    """
    console = Console(no_color=no_color, width=width)
    quiet = "" if no_color else "dim"

    if explain_name is not None:
        _run_explain(console, explain_name)
        return

    if path is None:
        typer.echo("error: a source file is required", err=True)
        raise typer.Exit(code=1)

    level = normalize_level(level)
    check_level("--level", level)
    _check_source(path)

    compiler = choose_compiler(compiler)
    levels = list(dict.fromkeys([BASELINE_LEVEL, level]))
    cleaned = {
        name: strip_directives(text)
        for name, text in compile_at_levels(path, compiler, levels).items()
    }

    baseline = _baseline_body(cleaned, func)
    if not baseline.strip():
        console.print(f"no functions to annotate in {path}", style=quiet)
        return

    # empty on the optimized side is a finding, not a mistake: the name was in
    # the baseline, so something the compiler did made it stop being anywhere
    optimized = find_function(cleaned[level], func)
    if not optimized.strip():
        console.print(f"the function is gone at -O{level} (inlined or optimized away)",
                      style=quiet)
        return

    annotations = find_annotations(baseline, optimized)
    if not summary:
        console.print(render_annotated(f"-O{level}", optimized, annotations, color=not no_color))
    _report(console, annotations, quiet)
