"""The diff command — compare the assembly of two optimization levels.

The command wiring is still the skeleton: it checks the source file and
reports which two levels it will compare. The rendering comes later, but
the line-diffing engine that feeds it lives here now (`diff_lines`).
"""

from difflib import SequenceMatcher
from pathlib import Path

import typer
from rich.text import Text

# the character we put in front of each line to say what happened to it,
# same idea as a normal `diff`/`git diff` gutter
GUTTER = {"add": "+", "remove": "-", "equal": " "}

# color for each kind of line once we're drawing to a real terminal. green for
# something that showed up and red for something that went away, matching what
# git diff does so it reads the way you'd expect. equal lines are just context
# so they keep the default color.
COLORS = {"add": "green", "remove": "red", "equal": ""}


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


def run_diff(path: Path, from_level: str = "0", to_level: str = "2") -> None:
    """Entry point for `compopt diff`.

    Eventually this compiles the file and shows what changed in the asm
    going from one -O level to another. Right now it only validates the
    input and prints what it's going to do, so the rest of the command
    can be built on top of a command that already exists and is wired in.
    """
    if not path.exists():
        typer.echo(f"error: no such file: {path}", err=True)
        raise typer.Exit(code=1)

    if not path.is_file():
        typer.echo(f"error: not a file: {path}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"would diff -O{from_level} against -O{to_level} for {path}")
