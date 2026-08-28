"""The same question `report.py` asks gcc, asked of clang instead.

clang has never had `-fopt-info-all` and doesn't pretend to — the flag comes
straight back as an unknown argument. What it has is `-Rpass`, three switches
that each take a regex over pass names and turn the matching passes' remarks
on:

    -Rpass=          passes that fired
    -Rpass-missed=   passes that wanted to fire and gave up
    -Rpass-analysis= what the analysis passes worked out on the way

which is close enough to gcc's optimized/missed/note that both ends can share
`OptRecord`. Two things are different enough to keep this out of `report.py`
rather than adding a branch to every function in it:

- the remarks come out on stderr, mixed in with the warnings, instead of into
  a file of their own. Telling the two apart is this module's job.
- clang names the pass that spoke. gcc leaves it in the wording, when it
  mentions it at all.

`passes.py` is what puts one report in front of both; this is the half that
gets the words out of clang.
"""

import re
import subprocess
import tempfile
from pathlib import Path

from compopt.compilers import CompileError
from compopt.report import MISSED, NOTE, OPTIMIZED, OptRecord

# a pass name regex that matches all of them, which makes these three together
# clang's answer to -fopt-info-all.
EVERY_PASS = ".*"
REMARK_FLAGS = (
    f"-Rpass={EVERY_PASS}",
    f"-Rpass-missed={EVERY_PASS}",
    f"-Rpass-analysis={EVERY_PASS}",
)

# clang prints the line of source under each diagnostic with a caret at the
# column. Good to read one at a time, but with every pass talking it's two
# lines of echo between every pair of remarks, so it goes off at the source
# rather than being filtered out again down here.
NO_CARET_FLAG = "-fno-caret-diagnostics"

# what goes at the end of every remark to say which switch turned it on:
#
#     ... [-Rpass=inline]
#     ... [-Rpass-missed=regalloc]
#     ... [-Rpass-analysis=size-info]
#
# so it carries both halves we can't get anywhere else — which of the three
# sorts this is, and the name of the pass that said it.
REMARK_TAG = re.compile(r"\s*\[-Rpass(?P<sort>-missed|-analysis)?=(?P<name>[^\]]+)\]$")

# the tag's middle bit, in gcc's vocabulary. an analysis remark is a pass
# thinking out loud rather than a decision either way, same as gcc's note.
SORTS = {None: OPTIMIZED, "-missed": MISSED, "-analysis": NOTE}

# any clang diagnostic, whatever sort. Used to find where one remark stops
# rather than to read it — a warning about the source is still a good place
# to say the remark before it has ended.
DIAGNOSTIC = re.compile(r"^.+?:\d+(?::\d+)?: (?P<kind>remark|warning|error|note): ")

# one remark, from its position to the end of whatever it had to say. DOTALL
# because the message is allowed to run over several lines and the whole thing
# has to come back as the message; see `remark_blocks`.
REMARK = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+)(?::(?P<column>\d+))?: remark: (?P<message>.*)$",
    re.DOTALL,
)


class RemarksUnsupported(Exception):
    """Raised when the compiler doesn't know what -Rpass is.

    The mirror of `report.OptInfoUnsupported`, and its own type for the same
    reason: the source is fine, we just asked a clang question of something
    that isn't clang. Real GNU gcc lands here every time.
    """

    def __init__(self, compiler: str) -> None:
        self.compiler = compiler
        super().__init__(f"{compiler} does not support -Rpass")


def rejected_the_flags(message: str) -> bool:
    """Whether a failed run was the compiler turning the flags down.

    gcc words it

        gcc: error: unrecognized command-line option '-Rpass=.*'

    and quotes the flag back, so looking for our own flag name covers it
    without having to know the wording. An error about the source names the
    source, not a flag we passed.
    """
    return "-Rpass" in message


def capture_remarks(source: Path, level: str, compiler: str) -> str:
    """Compile at one -O level and hand back everything the passes said.

    Same shape as `report.capture_opt_info` — throwaway temp dir, compile into
    it, drop the asm. The asm still has to be generated, because a pass that
    doesn't run has nothing to remark on.

    What comes back is stderr, warnings and all. There's no file to send the
    remarks to the way gcc has one: `-fsave-optimization-record` writes a YAML
    file, but that's a different flag with a different format, and the plan
    here is the -Rpass one. `parse_remarks` does the separating.

    Worth knowing before comparing the two: -O0 is empty for gcc and isn't for
    clang. Instructions still have to be selected and a frame still has to be
    laid out however few optimizations ran, and those passes report their work
    like any other. Nothing among them ever comes back `optimized`.
    """
    with tempfile.TemporaryDirectory(prefix="compopt-") as workdir:
        asm = Path(workdir) / "out.s"

        cmd = [
            compiler,
            "-S",
            f"-O{level}",
            *REMARK_FLAGS,
            NO_CARET_FLAG,
            str(source),
            "-o",
            str(asm),
        ]
        # no check=True, same as everywhere else: we want stderr in hand so it
        # can go into our own error instead of a CalledProcessError.
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = result.stderr or result.stdout or "compilation failed"
            if rejected_the_flags(detail):
                raise RemarksUnsupported(compiler)
            raise CompileError(compiler, detail)

        return result.stderr


def remark_blocks(text: str) -> list[str]:
    """Cut a run of stderr into one string per remark, wrapped ones kept whole.

    Most remarks are a line each, but not all of them — the asm-printer ones
    list the instructions in a basic block one per line and only get to their
    tag several lines down. Reading stderr a line at a time would take the
    first line of those as a remark with no tag and the rest as nothing at
    all, so lines are gathered up until the next diagnostic starts.

    Anything that isn't a remark ends the block it interrupts and starts
    nothing: a warning about the source is clang answering a different
    question, and belongs to whoever asked it.
    """
    blocks: list[list[str]] = []
    current: list[str] | None = None

    for line in text.splitlines():
        head = DIAGNOSTIC.match(line)
        if head is not None:
            if current is not None:
                blocks.append(current)
            current = [line] if head.group("kind") == "remark" else None
        elif current is not None:
            current.append(line)

    if current is not None:
        blocks.append(current)
    return ["\n".join(block) for block in blocks]


def parse_remark(block: str) -> OptRecord | None:
    """Turn one remark into a record, or None if it isn't one we can place.

    The tag at the end is what makes it readable — without it there's no way
    to say whether a pass fired or gave up, and guessing from the wording is
    exactly the reading-backwards this whole half of the tool exists to avoid.
    A remark that hasn't got one is dropped rather than filed under a sort we
    made up.
    """
    match = REMARK.match(block)
    if match is None:
        return None

    tag = REMARK_TAG.search(match.group("message"))
    if tag is None:
        return None

    column = match.group("column")
    return OptRecord(
        kind=SORTS[tag.group("sort")],
        message=match.group("message")[: tag.start()].strip(),
        file=match.group("file"),
        line=int(match.group("line")),
        column=None if column is None else int(column),
        pass_name=tag.group("name"),
    )


def parse_remarks(text: str) -> list[OptRecord]:
    """Every remark in a captured -Rpass run, in the order clang printed them.

    Same contract as `report.parse_opt_info`: order is the order the passes
    ran, and repeats are kept. clang repeats itself as readily as gcc does —
    the size-info pass reports a count for the module and then the same count
    again for the function, and both are true.
    """
    records = []
    for block in remark_blocks(text):
        record = parse_remark(block)
        if record is not None:
            records.append(record)
    return records
