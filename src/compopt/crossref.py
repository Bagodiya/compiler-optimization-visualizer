"""Lining the compiler's pass report up against the assembly it produced.

`report.py` gets records out of `-fopt-info-all`, but every one of them points
at a line of *C*. The renderer numbers lines of *asm*. Nothing joins the two
up, so a record can't be put beside the instructions it's actually about.

The join comes from the compiler itself. Built with `-g` it drops `.loc`
directives through the asm — "everything after me came from source line N" —
and that's a source-line-to-asm-line map already written down, no guessing
needed. This module reads it back and uses it to give each record the asm
lines it belongs to.

Two things to know about using it:

- `file_table` wants the whole asm, before `strip_directives` has been near
  it. The `.file` lines it reads are on the noise list and get thrown away.
- `strip_debug_lines` wants the body you are about to display, after
  isolating the function, because the numbers it hands back are counted the
  way `render.line_number_gutter` counts them.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from compopt.report import OptRecord

# the numbered form, `.file 1 "loop.c"` on ELF and `.file 1 "/tmp" "loop.c"`
# on Mach-O, where the directory got split off into a string of its own. gcc
# also writes a plain unnumbered `.file "loop.c"` at the top of every file to
# name the primary source, and that one isn't part of the table, which is why
# the number is required here rather than optional.
FILE_DIRECTIVE = re.compile(r'^\s*\.file\s+(?P<number>\d+)\s+(?P<names>.+)$')
QUOTED = re.compile(r'"([^"]*)"')

# `.loc <file> <line> <column>`, with a tail of flags after it that we don't
# need — is_stmt, prologue_end, discriminator, and a trailing `## loop.c:4:16`
# comment on clang. Only the first two numbers get captured. The column is in
# there but deliberately ignored: gcc reports the column of the expression it
# optimized and the .loc names the column of whatever operand is being loaded,
# so they disagree constantly on lines that do match.
LOC_DIRECTIVE = re.compile(r"^\s*\.loc\s+(?P<file>\d+)\s+(?P<line>\d+)")

# clang with -g also scatters these through the body. They're comments, so
# strip_directives leaves them alone, but they're only there because we asked
# for debug info and they'd be noise in the output.
DEBUG_VALUE = "##DEBUG_VALUE:"

# `.loc` uses line 0 for instructions the compiler made up itself and can't
# blame on any line of C — spill code, loop scaffolding, the prologue. No
# record will ever point there, so those runs are dropped.
NO_SOURCE_LINE = 0


def file_table(asm: str) -> dict[int, str]:
    """Read the `.file` directives into {file number: file name}.

    `.loc` refers to its file by number, so without this a record from a
    header and a record from the .c file are indistinguishable once they're
    both down to "line 3".

    Only the last quoted string on the line is kept, which is the file part
    in both spellings: Mach-O splits the directory off into a string of its
    own and puts it first, ELF writes the one string it was given. So an ELF
    entry can come back as a whole path where a Mach-O one is a bare name.
    That's fine either way — `cross_reference` compares on the name.
    """
    table = {}
    for line in asm.splitlines():
        match = FILE_DIRECTIVE.match(line)
        if match is None:
            continue
        names = QUOTED.findall(match.group("names"))
        if names:
            table[int(match.group("number"))] = names[-1]
    return table


def strip_debug_lines(body: str) -> tuple[str, dict[int, tuple[int, int]]]:
    """Take the debug bookkeeping out of a body, keeping what it told us.

    Hands back the body without it, plus a map from line number in *that*
    body to the (file number, source line) the compiler said it came from.
    Both halves are needed together: dropping the directives shifts every
    line below them, so a map built before the drop points at the wrong
    instructions afterwards.

    Lines with no `.loc` above them yet aren't in the map at all — the label
    a function opens with sits before the first one — and neither are the
    line-0 runs.
    """
    kept: list[str] = []
    origin: dict[int, tuple[int, int]] = {}
    current: tuple[int, int] | None = None

    for line in body.splitlines():
        match = LOC_DIRECTIVE.match(line)
        if match is not None:
            source_line = int(match.group("line"))
            if source_line == NO_SOURCE_LINE:
                current = None
            else:
                current = (int(match.group("file")), source_line)
            continue
        if line.strip().startswith(DEBUG_VALUE):
            continue
        kept.append(line)
        if current is not None:
            origin[len(kept)] = current

    return "\n".join(kept), origin


def _runs(lines: tuple[int, ...]) -> list[tuple[int, int]]:
    """Fold a sorted run of line numbers into (first, last) pairs."""
    ranges: list[tuple[int, int]] = []
    for line in lines:
        if ranges and line == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], line)
        else:
            ranges.append((line, line))
    return ranges


@dataclass(frozen=True, slots=True)
class LocatedRecord:
    """One report record and the asm lines it turned out to be about.

    `lines` is every asm line the compiler traced back to this record's
    source line, in order, and it is allowed to be empty. That isn't a
    failure to look hard enough — a missed pass is the compiler saying it
    generated nothing there, and a note about the function as a whole has no
    single line to sit on. Whatever prints these has to say so rather than
    pretend the record wasn't reported.
    """

    record: OptRecord
    lines: tuple[int, ...] = ()

    @property
    def found(self) -> bool:
        """Whether any asm was traced back to this record."""
        return bool(self.lines)

    @property
    def start(self) -> int | None:
        """First asm line, or None if nothing matched."""
        return self.lines[0] if self.lines else None

    @property
    def end(self) -> int | None:
        """Last asm line, or None if nothing matched."""
        return self.lines[-1] if self.lines else None

    def runs(self) -> list[tuple[int, int]]:
        """The matched lines as contiguous (first, last) ranges.

        One line of C rarely comes out as one block of asm. An unrolled loop
        body alternates between two source lines the whole way down, so line 4
        lands in a dozen separate places and `start`..`end` would claim the
        lot, most of it belonging to line 3. The ranges are what's really
        there; `start` and `end` are the outer bounds and no more than that.
        """
        return _runs(self.lines)


def _same_file(name: str, wanted: str) -> bool:
    """Whether two file names name the same file.

    Compared on the name alone. The compiler writes the path the way it was
    handed to it on the command line, and the report and the `.file` entry
    don't always get handed the same spelling — one absolute, one relative to
    wherever the compile ran.
    """
    return Path(name).name == Path(wanted).name


def cross_reference(
    records: list[OptRecord],
    origin: dict[int, tuple[int, int]],
    files: dict[int, str] | None = None,
) -> list[LocatedRecord]:
    """Give every record the asm lines that came from its source line.

    `origin` is the second half of what `strip_debug_lines` returned and
    `files` the table from `file_table`. Order is kept, so the result reads in
    the order the passes ran, same as `parse_opt_info` leaves them.

    An empty or missing `files` turns the file check off and matches on the
    line number alone, which is what a single-file example wants anyway. With
    a table in hand a `.loc` naming a number that isn't in it matches nothing:
    a table that has been read from the whole asm knows every file the asm
    mentions, so a number missing from it means the table came from somewhere
    it shouldn't have, and hanging a note off the wrong instruction is a worse
    answer than hanging it off none.
    """
    located = []
    for record in records:
        lines = tuple(
            asm_line
            for asm_line, (number, source_line) in sorted(origin.items())
            if source_line == record.line
            and (not files or _same_file(files.get(number, ""), record.file))
        )
        located.append(LocatedRecord(record, lines))
    return located
