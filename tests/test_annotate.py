"""Tests for the annotate command and the Annotation type.

No detectors exist yet, so these build annotations by hand and check the
range bookkeeping — the part every detector is going to depend on. The
command is still a skeleton too, so those tests only check it validates the
file it was handed.
"""

import dataclasses
from pathlib import Path

import pytest
from typer.testing import CliRunner

from compopt.annotate import Annotation
from compopt.cli import app

runner = CliRunner()


def test_fields_round_trip() -> None:
    note = Annotation("constant folding", 3, 5, "the arithmetic was done at compile time")
    assert note.name == "constant folding"
    assert note.start == 3
    assert note.end == 5
    assert note.description == "the arithmetic was done at compile time"


def test_description_is_optional() -> None:
    note = Annotation("dead code elimination", 1, 2)
    assert note.description == ""


def test_span_counts_both_ends() -> None:
    assert Annotation("inlining", 4, 4).span == 1
    assert Annotation("inlining", 4, 9).span == 6


def test_covers_is_inclusive() -> None:
    note = Annotation("loop unrolling", 10, 14)
    assert note.covers(10)
    assert note.covers(12)
    assert note.covers(14)
    assert not note.covers(9)
    assert not note.covers(15)


def test_label_uses_singular_for_one_line() -> None:
    assert Annotation("tail call", 7, 7).label() == "tail call (line 7)"


def test_label_uses_a_range_for_several_lines() -> None:
    assert Annotation("vectorization", 7, 11).label() == "vectorization (lines 7-11)"


def test_start_below_one_is_rejected() -> None:
    # line numbers start at 1 in the gutter, so 0 means someone passed a
    # list index straight through without adjusting it
    with pytest.raises(ValueError, match="start must be 1 or greater"):
        Annotation("branch elimination", 0, 3)


def test_backwards_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="comes before start"):
        Annotation("strength reduction", 8, 5)


def test_blank_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="needs a name"):
        Annotation("   ", 1, 2)


def test_annotation_is_frozen() -> None:
    note = Annotation("register coalescing", 2, 3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        note.name = "something else"


def test_annotate_names_the_file(tmp_path: Path) -> None:
    src = tmp_path / "hello.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    result = runner.invoke(app, ["annotate", str(src)])
    assert result.exit_code == 0
    # the placeholder should at least say which file it would work on
    assert "hello.c" in result.stdout


def test_annotate_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.c"

    result = runner.invoke(app, ["annotate", str(missing)])
    assert result.exit_code == 1


def test_annotate_directory_is_rejected(tmp_path: Path) -> None:
    # a directory isn't a source file, so this fails like a missing one
    result = runner.invoke(app, ["annotate", str(tmp_path)])
    assert result.exit_code == 1


def test_annotate_shows_up_in_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "annotate" in result.stdout
