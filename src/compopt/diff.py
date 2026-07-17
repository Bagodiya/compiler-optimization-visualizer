"""The diff command — compare the assembly of two optimization levels.

The command wiring is still the skeleton: it checks the source file and
reports which two levels it will compare. The rendering comes later, but
the line-diffing engine that feeds it lives here now (`diff_lines`).
"""

from difflib import SequenceMatcher
from pathlib import Path

import typer
from rich.text import Text

from compopt.compilers import DEFAULT_LEVELS

# the character we put in front of each line to say what happened to it,
# same idea as a normal `diff`/`git diff` gutter. "gap" is our stand-in for a
# run of unchanged lines we folded away, so it gets the @@ marker git uses for
# its hunk headers.
GUTTER = {"add": "+", "remove": "-", "equal": " ", "gap": "@@"}

# color for each kind of line once we're drawing to a real terminal. green for
# something that showed up and red for something that went away, matching what
# git diff does so it reads the way you'd expect. equal lines are just context
# so they keep the default color, and the folded-away marker is cyan like a
# git hunk header.
COLORS = {"add": "green", "remove": "red", "equal": "", "gap": "cyan"}


def diff_lines(old: str, new: str) -> list[tuple[str, str]]:
    """Line-by-line diff between two blocks of assembly.

    Returns a flat list of (tag, line) pairs in the order they should be
    shown, where tag is one of:

    - "equal"  the line is the same in both
    - "remove" the line is only in `old` (went away in `new`)
    - "add"    the line is only in `new` (showed up going from old to new)

    We lean on difflib's SequenceMatcher to find the matching runs. A
    "replace" chunk (lines that differ on both sides) is just emitted as
    the removals first, then the additions, which is what a normal diff
    looks like anyway. The rendering step turns these tags into +/- later.
    """
    old_lines = old.splitlines()
    new_lines = new.splitlines()

    matcher = SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    result: list[tuple[str, str]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            result.extend(("equal", line) for line in old_lines[i1:i2])
        else:
            # replace/delete/insert: show what left, then what arrived
            result.extend(("remove", line) for line in old_lines[i1:i2])
            result.extend(("add", line) for line in new_lines[j1:j2])

    return result


def _gap_text(hidden: int) -> str:
    # the little summary that stands in for the lines we folded away
    plural = "" if hidden == 1 else "s"
    return f"{hidden} unchanged line{plural}"


def trim_context(diff: list[tuple[str, str]], context: int) -> list[tuple[str, str]]:
    """Keep only `context` unchanged lines on either side of each change.

    A full asm diff is mostly lines that didn't move, so the interesting
    parts get lost in the noise. This keeps up to `context` equal lines next
    to anything that was added or removed and folds the rest of the equal
    runs into a single "gap" line that just says how many lines were hidden,
    the same way `diff -U` trims its context.

    A negative `context` means "don't trim", so the diff comes back untouched.
    """
    if context < 0:
        return list(diff)

    # first figure out which lines are close enough to a change to keep. a
    # changed line always counts, and so does anything within `context` of it.
    keep = [False] * len(diff)
    for i, (tag, _) in enumerate(diff):
        if tag == "equal":
            continue
        lo = max(0, i - context)
        hi = min(len(diff), i + context + 1)
        for k in range(lo, hi):
            keep[k] = True

    # now walk the diff, passing kept lines through and collapsing each run of
    # dropped lines into one gap marker
    trimmed: list[tuple[str, str]] = []
    hidden = 0
    for i, entry in enumerate(diff):
        if keep[i]:
            if hidden:
                trimmed.append(("gap", _gap_text(hidden)))
                hidden = 0
            trimmed.append(entry)
        else:
            hidden += 1
    if hidden:
        trimmed.append(("gap", _gap_text(hidden)))

    return trimmed


def render_diff(diff: list[tuple[str, str]]) -> str:
    """Turn the (tag, line) pairs from `diff_lines` into text with a gutter.

    Every line gets a one-character marker in front of it: "+" for a line
    that showed up, "-" for one that went away, and a space for a line that
    stayed the same. That's the plain form you'd recognize from `diff`; the
    coloring on top of it comes later.
    """
    return "\n".join(f"{GUTTER[tag]} {line}" for tag, line in diff)


def highlight_diff(diff: list[tuple[str, str]], color: bool = True) -> Text:
    """Colored version of `render_diff` for showing on a terminal.

    Same +/- gutter as the plain form, but each line is tinted by what
    happened to it: green for an added line, red for a removed one, and no
    color for the lines that stayed the same. With ``color`` off we just wrap
    the plain text so piped output doesn't carry any escape codes.
    """
    if not color:
        return Text(render_diff(diff))
    out = Text()
    for i, (tag, line) in enumerate(diff):
        if i:
            out.append("\n")
        out.append(f"{GUTTER[tag]} {line}", style=COLORS[tag])
    return out


def _check_level(flag: str, level: str) -> None:
    """Stop early if a level isn't one we know how to compile.

    Only the digits in DEFAULT_LEVELS are valid, so `--from 9` is caught
    here instead of turning into a `-O9` the compiler would reject.
    """
    if level not in DEFAULT_LEVELS:
        typer.echo(f"error: {flag} must be one of: {', '.join(DEFAULT_LEVELS)}", err=True)
        raise typer.Exit(code=1)


def _check_context(context: int) -> None:
    """Reject a negative --context before we get any further.

    Zero is fine (show only the changed lines), but a negative count doesn't
    mean anything so we bail instead of silently treating it as "show all".
    """
    if context < 0:
        typer.echo("error: --context must be zero or greater", err=True)
        raise typer.Exit(code=1)


def run_diff(path: Path, from_level: str = "0", to_level: str = "2", context: int = 3) -> None:
    """Entry point for `compopt diff`.

    Eventually this compiles the file and shows what changed in the asm
    going from one -O level to another. Right now it only validates the
    input and prints what it's going to do, so the rest of the command
    can be built on top of a command that already exists and is wired in.

    The two levels are the bare digits ("0", "2"), and default to comparing
    -O0 against -O2 since that's the pair that shows the biggest change.
    `context` is how many unchanged lines to keep around each change once the
    real rendering is wired up (see `trim_context`).
    """
    # check the flags before touching the disk, they're cheap to get wrong
    _check_level("--from", from_level)
    _check_level("--to", to_level)
    _check_context(context)

    if not path.exists():
        typer.echo(f"error: no such file: {path}", err=True)
        raise typer.Exit(code=1)

    if not path.is_file():
        typer.echo(f"error: not a file: {path}", err=True)
        raise typer.Exit(code=1)

    typer.echo(
        f"would diff -O{from_level} against -O{to_level} for {path} "
        f"with {context} lines of context"
    )
