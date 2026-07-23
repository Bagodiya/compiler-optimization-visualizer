"""The diff command — compare the assembly of two optimization levels.

The command wiring is still the skeleton: it checks the source file and
reports which two levels it will compare. The rendering comes later, but
the line-diffing engine that feeds it lives here now (`diff_lines`).
"""

from difflib import SequenceMatcher
from difflib import unified_diff as _unified_diff
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

# what we print instead of the asm when the two levels came out the same
IDENTICAL_MESSAGE = "no difference: both levels produced the same assembly"

# and what we print when the function isn't in one (or either) of the levels
NEITHER_MESSAGE = "nothing to compare: the function is not in either level"
GONE_MESSAGE = "the function is gone at -O{to} (inlined or optimized away)"
NEW_MESSAGE = "the function only shows up at -O{to}, there is nothing at -O{frm}"


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


def is_identical(diff: list[tuple[str, str]]) -> bool:
    """True when the two levels compiled down to exactly the same asm.

    This happens more often than you'd think — -O2 and -O3 give the same
    output for plenty of small functions, and comparing a level against
    itself obviously does too. Printing a wall of unchanged lines there is
    just noise, so the renderers use this to print one line instead.

    An empty diff means there was nothing to compare in the first place
    (an empty function, say), which isn't the same thing, so that's False.
    """
    return bool(diff) and all(tag == "equal" for tag, _ in diff)


def missing_message(old: str, new: str, from_level: str = "0",
                    to_level: str = "2") -> str | None:
    """Explain an empty side, or None when there are two bodies to diff.

    A function that got inlined at -O2 simply isn't in that assembly any
    more, so `find_function` hands back "" for it. Diffing that against the
    -O0 body technically works, but every single line comes out marked "-",
    which reads like the optimizer deleted the code one instruction at a
    time instead of telling you what actually happened. One line saying the
    function is gone is a lot more useful.

    The other two empty cases are here too: nothing at the low level but
    something at the high one (rare, but static functions can go the other
    way), and nothing at either, which usually just means a bad --func.
    """
    if not old.strip() and not new.strip():
        return NEITHER_MESSAGE
    if not new.strip():
        return GONE_MESSAGE.format(to=to_level)
    if not old.strip():
        return NEW_MESSAGE.format(frm=from_level, to=to_level)
    return None


def render_diff(diff: list[tuple[str, str]]) -> str:
    """Turn the (tag, line) pairs from `diff_lines` into text with a gutter.

    Every line gets a one-character marker in front of it: "+" for a line
    that showed up, "-" for one that went away, and a space for a line that
    stayed the same. That's the plain form you'd recognize from `diff`; the
    coloring on top of it comes later.

    If nothing changed at all we say so in one line rather than echoing the
    whole function back with a blank gutter (see `is_identical`).
    """
    if is_identical(diff):
        return IDENTICAL_MESSAGE
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
    if is_identical(diff):
        # not a change, just a note about the run, so keep it dim
        return Text(IDENTICAL_MESSAGE, style="dim")
    out = Text()
    for i, (tag, line) in enumerate(diff):
        if i:
            out.append("\n")
        out.append(f"{GUTTER[tag]} {line}", style=COLORS[tag])
    return out


def unified_diff(old: str, new: str, from_label: str = "O0", to_label: str = "O2",
                 context: int = 3) -> str:
    """Render the change in the plain unified-diff format (`diff -u`/`git diff`).

    Our own +/- gutter (`render_diff`) is easy to read but you can't feed it to
    `patch` or paste it into a review tool. This is the portable version: the
    familiar `--- old` / `+++ new` header followed by `@@ -a,b +c,d @@` hunks
    with `context` unchanged lines kept around each run of changes.

    I'm letting difflib do the hunk math here rather than reusing `diff_lines`,
    since getting the line numbers in the `@@` headers right by hand is exactly
    the kind of thing the stdlib already gets right. The labels name the two
    sides, and we tag them with the -O level they came from so the header says
    what's being compared.
    """
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    lines = _unified_diff(
        old_lines,
        new_lines,
        fromfile=from_label,
        tofile=to_label,
        n=context,
        lineterm="",
    )
    return "\n".join(lines)


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


def run_diff(path: Path, from_level: str = "0", to_level: str = "2", context: int = 3,
             unified: bool = False) -> None:
    """Entry point for `compopt diff`.

    Eventually this compiles the file and shows what changed in the asm
    going from one -O level to another. Right now it only validates the
    input and prints what it's going to do, so the rest of the command
    can be built on top of a command that already exists and is wired in.

    The two levels are the bare digits ("0", "2"), and default to comparing
    -O0 against -O2 since that's the pair that shows the biggest change.
    `context` is how many unchanged lines to keep around each change once the
    real rendering is wired up (see `trim_context`). When `unified` is set the
    output uses the portable `diff -u` format (see `unified_diff`) instead of
    our colored +/- view.
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

    message = (
        f"would diff -O{from_level} against -O{to_level} for {path} "
        f"with {context} lines of context"
    )
    if unified:
        message += ", in unified format"
    typer.echo(message)
