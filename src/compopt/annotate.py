"""The annotate command: compile a file, run the detectors, print what they say.

The detecting itself lives in `compopt.detectors`, one optimization per module.
What's left here is the command around them — working out which two levels to
compile, pulling the wanted function out of each, and printing the result.

`--report` is the same question asked the other way round. Instead of reading
the asm and working backwards, it asks the compiler what it did and prints the
answer. `report.py` gets that answer out, `crossref.py` works out which lines
of asm each part of it is about, and the printing is down at the bottom here.
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.text import Text

from compopt.annotation import Annotation
from compopt.asm import find_function, function_names, isolate_function, strip_directives
from compopt.compilers import (
    check_level,
    choose_compiler,
    compile_at_levels,
    compile_to_asm,
    normalize_level,
)
from compopt.crossref import (
    LocatedRecord,
    cross_reference,
    file_table,
    strip_debug_lines,
)
from compopt.detectors import DESCRIPTIONS, find_annotations, match_name
from compopt.render import render_annotated
from compopt.report import (
    MISSED,
    NOTE,
    OPT_INFO_FLAG,
    OPTIMIZED,
    OptInfoUnsupported,
    capture_opt_info,
    parse_opt_info,
)

# the level everything is compared against. the detectors that need two bodies
# need one where the compiler hasn't done anything yet, and that's -O0.
BASELINE_LEVEL = "0"

# colors for the three things -fopt-info can say. a pass that fired is the good
# news, a missed one is the compiler pointing at where it gave up, and the notes
# in between are running commentary and shouldn't shout.
REPORT_STYLES = {OPTIMIZED: "green", MISSED: "yellow", NOTE: "cyan"}


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


def _isolate_or_stop(cleaned: str, func: str | None) -> str:
    """Pull one function out of a cleaned body, or stop saying what is there.

    A name that isn't in the file is a typo, and the useful thing to print is
    the list of names that are, so you can see the one you meant.
    """
    try:
        return isolate_function(cleaned, func)
    except KeyError:
        names = function_names(cleaned)
        typer.echo(f"error: no function named {func!r}", err=True)
        if names:
            typer.echo(f"available functions: {', '.join(names)}", err=True)
        raise typer.Exit(code=1) from None


def _baseline_body(cleaned: dict[str, str], func: str | None) -> str:
    """The -O0 body of the wanted function, or a clean error naming what's there.

    The baseline is where a name has to exist, because -O0 does what the source
    said. Anything missing from it was never written rather than optimized out,
    so a miss here is a typo and gets reported as one.
    """
    return _isolate_or_stop(cleaned[BASELINE_LEVEL], func)


def _asm_range(found: LocatedRecord) -> str:
    """Say which asm lines a record came out as, in the fewest words it takes.

    One source line usually turns into several separate runs of asm rather than
    one block, so the ranges get listed out instead of collapsed to first..last
    — see `LocatedRecord.runs`. Nothing matching is a real answer too: a missed
    pass generated no code by definition, and a note about the whole function
    has no one line to sit on.
    """
    if not found.found:
        return "no asm came from this line"
    spans = ", ".join(
        str(first) if first == last else f"{first}-{last}"
        for first, last in found.runs()
    )
    plural = "" if len(found.lines) == 1 else "s"
    return f"asm line{plural} {spans}"


def _report_line(found: LocatedRecord, color: bool) -> Text:
    """One record as a line of output: where it was, what kind, what it said.

    Built as a `Text` rather than handed to `console.print` as a string because
    the message is the compiler's, and rich would read any square brackets in
    it as markup of ours.
    """
    record = found.record
    line = Text("  ")
    line.append(record.where(), style="dim" if color else "")
    line.append("  ")
    line.append(f"{record.kind}: ", style=REPORT_STYLES[record.kind] if color else "")
    line.append(record.message)
    return line


def _print_report(console: Console, located: list[LocatedRecord], level: str,
                  compiler: str, quiet: str, color: bool) -> None:
    """Print everything the compiler said, in the order its passes said it."""
    if not located:
        # -O0 lands here honestly: no passes ran, so there was nothing to say
        console.print(f"{compiler} reported nothing at -O{level}", style=quiet)
        return

    plural = "" if len(located) == 1 else "s"
    console.print(f"{compiler} reported {len(located)} thing{plural} at -O{level}:\n")
    for found in located:
        console.print(_report_line(found, color))
        console.print(f"    {_asm_range(found)}", style=quiet)


def _run_report(console: Console, path: Path, level: str, func: str | None,
                compiler: str, quiet: str, color: bool) -> None:
    """Answer `--report`: print what the compiler said, not what we worked out.

    Two compiles, because the two halves come from different runs. One with
    `-fopt-info-all` for the words, one with `-g` for the `.loc` directives
    that say which asm each of those words is about. They're the same code
    either way — neither flag changes what gets generated.

    The report covers the whole file while the asm is one function, so records
    about the rest of it still get printed and come out with no asm against
    them. Dropping them would be tidier and would also hide half of what the
    compiler said, which is the opposite of what this flag is for.
    """
    try:
        text = capture_opt_info(path, level, compiler)
    except OptInfoUnsupported as err:
        typer.echo(f"error: {compiler} does not understand {OPT_INFO_FLAG}", err=True)
        typer.echo("that flag is GNU gcc's; on macOS `gcc` is normally Apple clang",
                   err=True)
        raise typer.Exit(code=1) from err

    asm = compile_to_asm(path, level, compiler, debug=True)
    body = _isolate_or_stop(strip_directives(asm), func)
    _, origin = strip_debug_lines(body)

    located = cross_reference(parse_opt_info(text), origin, file_table(asm))
    _print_report(console, located, level, compiler, quiet, color)


def run_annotate(path: Path | None, level: str = "2", func: str | None = None,
                 summary: bool = False, explain_name: str | None = None,
                 no_color: bool = False, width: int | None = None,
                 compiler: str | None = None, report: bool = False) -> None:
    """Entry point for `compopt annotate`.

    Compiles the file at -O0 and at the level asked for, runs every detector
    over the pair, and prints the optimized asm with what they found beside it
    and a list underneath. `summary` drops the asm and keeps just the list.

    `explain_name` is the odd one out — it looks up what an optimization means
    by name and prints that, without compiling anything, so it answers "what is
    register coalescing" as well as "what happened to this file". That's why
    the path is optional here.

    `report` takes the other route to the same question: rather than run the
    detectors, it asks the compiler for its own pass report and prints that.
    The two disagree fairly often, and that's the interesting part — a detector
    can only see the shape of the finished code, so it misses passes that left
    no trace and it names things the compiler would name differently. Neither
    list is the whole truth on its own. `summary` means nothing here; the report
    is a list already.

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

    if report:
        _run_report(console, path, level, func, compiler, quiet, not no_color)
        return

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
