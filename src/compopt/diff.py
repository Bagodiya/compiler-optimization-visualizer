"""The diff command — compare the assembly of two optimization levels."""

from difflib import SequenceMatcher
from difflib import unified_diff as _unified_diff
from pathlib import Path

import typer
from rich.console import Console
from rich.text import Text

from compopt.asm import find_function, strip_directives
from compopt.compilers import (
    check_level,
    choose_compiler,
    compile_at_levels,
    normalize_level,
)

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


def _check_context(context: int) -> None:
    """Reject a negative --context before we get any further.

    Zero is fine (show only the changed lines), but a negative count doesn't
    mean anything so we bail instead of silently treating it as "show all".
    """
    if context < 0:
        typer.echo("error: --context must be zero or greater", err=True)
        raise typer.Exit(code=1)


def run_diff(path: Path, from_level: str = "0", to_level: str = "2", context: int = 3,
             unified: bool = False, func: str | None = None, no_color: bool = False,
             width: int | None = None, compiler: str | None = None) -> None:
    """Entry point for `compopt diff`.

    Compiles the file at the two levels asked for, pulls the same function out
    of each, and shows what changed between them. The levels default to -O0
    against -O2 since that's the pair with the biggest change in it.

    A function that isn't in one of the two builds doesn't get diffed — that's
    the inlining case and `missing_message` says so in a line instead of
    marking the whole body as deleted. `context` trims the unchanged lines
    down to a few either side of each change, and `unified` swaps our colored
    +/- view for the portable `diff -u` format.

    Unlike `show` this compiles two levels rather than all four, since the
    other two would be work nobody asked to see.
    """
    # check the flags before touching the disk, they're cheap to get wrong
    from_level = normalize_level(from_level)
    to_level = normalize_level(to_level)
    check_level("--from", from_level)
    check_level("--to", to_level)
    _check_context(context)

    if not path.exists():
        typer.echo(f"error: no such file: {path}", err=True)
        raise typer.Exit(code=1)

    if not path.is_file():
        typer.echo(f"error: not a file: {path}", err=True)
        raise typer.Exit(code=1)

    compiler = choose_compiler(compiler)

    # the same level twice is one compile, not two — asking for it is a fair
    # way to check the tool agrees a level matches itself
    levels = list(dict.fromkeys([from_level, to_level]))
    asm = compile_at_levels(path, compiler, levels)

    old = find_function(strip_directives(asm[from_level]), func)
    new = find_function(strip_directives(asm[to_level]), func)

    console = Console(no_color=no_color, width=width)

    note = missing_message(old, new, from_level, to_level)
    if note is not None:
        console.print(note, style="" if no_color else "dim")
        return

    # soft_wrap because a diff line means the instruction it came from. folding
    # a long one onto a second row puts text under the +/- gutter that has no
    # marker of its own, which reads as a line that changed for free; letting
    # it run off the edge is what `diff` and `git diff` do
    if unified:
        console.print(unified_diff(old, new, f"O{from_level}", f"O{to_level}", context),
                      soft_wrap=True)
        return

    changes = diff_lines(old, new)
    # trimming turns a run of unchanged lines into a "gap" entry, which stops
    # `is_identical` recognising a body that didn't change at all, so the
    # question gets asked while the diff is still whole
    if not is_identical(changes):
        changes = trim_context(changes, context)
    console.print(highlight_diff(changes, color=not no_color), soft_wrap=True)
