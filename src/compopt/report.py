"""Asking the compiler what it did, instead of working it out from the asm.

Everything in `detectors/` reads the finished assembly and reasons backwards:
the frame pointer is gone, so the frame must have been elided. gcc doesn't
need to be guessed at like that. `-fopt-info-all` makes it say so directly —
every pass that fired, every one that wanted to fire and couldn't, each with
the source line it was looking at.

The two disagree often enough to be interesting, which is the whole point of
the report. This module gets the text out and turns it into records; lining
those up against the asm is the next step.
"""

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from compopt.compilers import CompileError

# gcc's "tell me what every optimization pass decided" switch. There's also
# -fopt-info-optimized for just the successes, but missed passes are the
# useful half — knowing gcc wanted to vectorize a loop and gave up says more
# than the asm ever will.
OPT_INFO_FLAG = "-fopt-info-all"


class OptInfoUnsupported(Exception):
    """Raised when the compiler doesn't know what -fopt-info-all is.

    Its own type rather than another CompileError, because nothing is wrong
    with the source — we asked a gcc question of something that isn't gcc.
    clang answers the same question through -Rpass instead, and on macOS the
    `gcc` on PATH is normally Apple clang under gcc's name, so this is the
    everyday case there and not some corner to apologise for.
    """

    def __init__(self, compiler: str) -> None:
        self.compiler = compiler
        super().__init__(f"{compiler} does not support {OPT_INFO_FLAG}")


def rejected_the_flag(message: str) -> bool:
    """Whether a failed run was the compiler turning down the flag itself.

    The two wordings are

        clang: error: unknown argument: '-fopt-info-all=/tmp/...'
        gcc: error: unrecognized command-line option '-fopt-info-all=/tmp/...'

    and both quote the flag back, so looking for our own flag in the message
    covers both and keeps working if a third compiler words it a fourth way.
    A genuine compile error is about the source file and won't name a flag we
    passed.
    """
    return OPT_INFO_FLAG in message


def capture_opt_info(source: Path, level: str, compiler: str) -> str:
    """Compile at one -O level and hand back the pass report as plain text.

    Same shape as `compile_to_asm` — throwaway temp dir, compile into it, read
    the result back, let the dir go. The asm gets written as well even though
    it's dropped here, because gcc won't report on passes it didn't run and it
    only runs them when it's really generating code.

    Sending the report to a file rather than letting it default to stderr is
    what keeps it clean: warnings from the source go to stderr too, and picking
    the two apart afterwards means guessing which lines were whose.
    """
    with tempfile.TemporaryDirectory(prefix="compopt-") as workdir:
        info = Path(workdir) / "opt-info.txt"
        asm = Path(workdir) / "out.s"

        cmd = [
            compiler,
            "-S",
            f"-O{level}",
            f"{OPT_INFO_FLAG}={info}",
            str(source),
            "-o",
            str(asm),
        ]
        # same reason as compile_to_asm: no check=True, we want stderr in hand
        # so it can go into our own error instead of a CalledProcessError.
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = result.stderr or result.stdout or "compilation failed"
            if rejected_the_flag(detail):
                raise OptInfoUnsupported(compiler)
            raise CompileError(compiler, detail)

        # gcc makes the file even with nothing to put in it, but a compiler
        # that accepted the flag and then wrote nowhere shouldn't blow up here.
        # -O0 lands on the empty string legitimately: no passes, no report.
        return info.read_text() if info.exists() else ""


# what gcc puts before the colon to say how the pass went. "optimized" is a
# pass that fired, "missed" is one that wanted to and gave up, and "note" is
# everything else it decided was worth mentioning on the way. they're the only
# three -fopt-info knows about, so anything else on a line means we misread it.
OPTIMIZED = "optimized"
MISSED = "missed"
NOTE = "note"
KINDS = (OPTIMIZED, MISSED, NOTE)

# a reported line looks like
#
#     loop.c:3:23: optimized: loop vectorized using 16 byte vectors
#     missed.c:2:23: missed: couldn't vectorize loop
#     loop.c:10:40: optimized:  Inlining helper/2 into caller/3.
#
# file, line, column, kind, message. the column is optional because older gcc
# leaves it off, and the file part is non-greedy so a path with colons in it
# doesn't eat the line number. the double space in the inlining message is
# gcc's, not a typo, which is why the message is stripped afterwards.
REPORT_LINE = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+)(?::(?P<column>\d+))?: "
    r"(?P<kind>optimized|missed|note): *(?P<message>.*)$"
)


@dataclass(frozen=True, slots=True)
class OptRecord:
    """One thing the compiler said about one place in the source.

    - ``kind`` which of `KINDS` it was
    - ``message`` what it said, with gcc's leading spaces taken off
    - ``file``, ``line``, ``column`` where in the *source* it was looking.
      Note that's the .c file, not the asm — the asm line these belong next
      to is what step 61 has to work out.

    The column is None when the compiler didn't give one. Frozen for the same
    reason `Annotation` is: it's a record of something already said, and
    nothing downstream should be editing the compiler's words.
    """

    kind: str
    message: str
    file: str
    line: int
    column: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.kind!r} is not one of {KINDS}")
        if self.line < 1:
            raise ValueError(f"line must be 1 or greater, got {self.line}")

    @property
    def helped(self) -> bool:
        """Whether this one is a pass that actually fired."""
        return self.kind == OPTIMIZED

    def where(self) -> str:
        """The source position back in gcc's own `file:line:col` spelling."""
        if self.column is None:
            return f"{self.file}:{self.line}"
        return f"{self.file}:{self.line}:{self.column}"


def parse_record(line: str) -> OptRecord | None:
    """Turn one line of the report into a record, or None if it isn't one.

    -fopt-info-all mixes the located reports in with the passes' own running
    commentary — "BB 3 is always executed in loop 1", "Unit growth for small
    function inlining: 16->16 (0%)", blank lines between phases. None of that
    names a source position, so there's nowhere to hang it in the output and
    nothing for step 61 to match it against. Those come back as None and the
    caller drops them.
    """
    match = REPORT_LINE.match(line)
    if match is None:
        return None

    column = match.group("column")
    return OptRecord(
        kind=match.group("kind"),
        message=match.group("message").strip(),
        file=match.group("file"),
        line=int(match.group("line")),
        column=None if column is None else int(column),
    )


def parse_opt_info(text: str) -> list[OptRecord]:
    """Every located report in a captured -fopt-info-all run, in order.

    Kept in the order gcc printed them, which is the order the passes ran, and
    kept whole — gcc says the same thing several times over when a pass retries
    a loop with different vector widths, and the repeats are how you tell it
    tried hard. Anything that wants one line per finding can fold them later.
    """
    records = []
    for line in text.splitlines():
        record = parse_record(line)
        if record is not None:
            records.append(record)
    return records
