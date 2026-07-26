"""Tests for the Annotation type.

No detectors exist yet, so these build annotations by hand and check the
range bookkeeping — the part every detector is going to depend on.
"""

import dataclasses

import pytest

from compopt.annotate import Annotation


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
