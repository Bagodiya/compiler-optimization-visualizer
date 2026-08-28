"""One pass report, whichever of the two compilers is answering.

`report.py` asks gcc through `-fopt-info-all` and `remarks.py` asks clang
through `-Rpass`. Both hand back `OptRecord`s, so the records themselves
already line up; what was still missing was anything that picks between the
two, which is why `--report` only ever worked on a real gcc.

Picking on the compiler's name doesn't work. `gcc` on macOS is Apple clang
wearing gcc's name, `cc` could be either, and `$CC` can point at anything at
all. So nothing here looks at the name: it asks one way, and if the compiler
turns the flag down it asks the other. A rejected flag is the one answer that
can't be a guess.
"""

from dataclasses import dataclass
from pathlib import Path

from compopt.remarks import RemarksUnsupported, capture_remarks, parse_remarks
from compopt.report import (
    OPT_INFO_FLAG,
    OptInfoUnsupported,
    OptRecord,
    capture_opt_info,
    parse_opt_info,
)

# `-Rpass` with the name left off. The three real flags carry a pass-name
# regex each (see `remarks.REMARK_FLAGS`) and none of them is the name of the
# thing as a whole, which is what output saying where the words came from
# wants.
RPASS_FLAG = "-Rpass"

# the two ways of asking, and the flag each one goes through
OPT_INFO = "opt-info"
REMARKS = "remarks"
FLAGS = {OPT_INFO: OPT_INFO_FLAG, REMARKS: RPASS_FLAG}


class NoPassReport(Exception):
    """Raised when a compiler answers to neither way of asking.

    Not the same thing as either `OptInfoUnsupported` or `RemarksUnsupported`
    on its own — each of those is the everyday case for the other compiler and
    means nothing more than "wrong one, try the other". Both of them together
    is the real dead end, and it's a third compiler under a name we know.
    """

    def __init__(self, compiler: str) -> None:
        self.compiler = compiler
        super().__init__(
            f"{compiler} supports neither {OPT_INFO_FLAG} nor {RPASS_FLAG}"
        )


@dataclass(frozen=True, slots=True)
class PassReport:
    """What one compiler said about one file at one -O level.

    `via` is which of the two ways got an answer, and it's kept rather than
    thrown away once the records are out: the two compilers don't say the same
    things, so output that doesn't name where the words came from is output you
    can't compare against anything. `flag` turns it back into the flag to
    print.

    `records` is a tuple rather than a list to keep the whole thing frozen,
    for the reason `OptRecord` is frozen — it's a record of something already
    said, and nothing downstream should be editing the compiler's words.
    """

    compiler: str
    level: str
    via: str
    records: tuple[OptRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.via not in FLAGS:
            raise ValueError(f"{self.via!r} is not one of {tuple(FLAGS)}")

    @property
    def flag(self) -> str:
        """The flag the records came out of, for saying so in the output."""
        return FLAGS[self.via]

    def of_kind(self, kind: str) -> list[OptRecord]:
        """Just the records of one kind, in the order the passes said them."""
        return [record for record in self.records if record.kind == kind]


def kind_label(record: OptRecord) -> str:
    """How a record's kind is spelled in the output, pass name and all.

    The kind on its own is all gcc gives us. clang names the pass behind every
    remark, and that name is the most useful part of what it says — "missed"
    tells you something gave up, "missed (loop-vectorize)" tells you what. So
    it goes in when it's there and the line reads the same either way when it
    isn't.
    """
    if record.pass_name is None:
        return record.kind
    return f"{record.kind} ({record.pass_name})"


def collect_report(source: Path, level: str, compiler: str) -> PassReport:
    """Compile once and give back whatever the compiler had to say about it.

    gcc's way is tried first and clang's second, though the order doesn't
    matter much: real gcc rejects `-Rpass` and real clang rejects
    `-fopt-info-all`, so at most one of them was ever going to answer.

    Falling through costs a wasted compile on clang, since the flag isn't
    turned down until the compiler has been started. Cheaper would be to probe
    with `--version` first, but that's a second process either way and it puts
    a guess where there doesn't have to be one.

    A `CompileError` isn't caught here on purpose. Source that doesn't build
    doesn't build for the other compiler either, and compiling it a second
    time to be told so again helps nobody.
    """
    try:
        text = capture_opt_info(source, level, compiler)
    except OptInfoUnsupported:
        pass
    else:
        return PassReport(compiler, level, OPT_INFO, tuple(parse_opt_info(text)))

    try:
        text = capture_remarks(source, level, compiler)
    except RemarksUnsupported as err:
        raise NoPassReport(compiler) from err

    return PassReport(compiler, level, REMARKS, tuple(parse_remarks(text)))
