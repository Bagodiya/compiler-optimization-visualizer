"""Tests for compiler detection."""

import pytest

from compopt import compilers
from compopt.compilers import find_compilers


def test_returns_subset_of_known() -> None:
    # whatever we get back has to be names we actually know about
    result = find_compilers()
    for name in result:
        assert name in compilers.KNOWN_COMPILERS


def test_no_compilers_when_path_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    # pretend nothing is installed
    monkeypatch.setattr(compilers.shutil, "which", lambda _name: None)
    assert find_compilers() == []


def test_only_gcc_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        compilers.shutil,
        "which",
        lambda name: "/usr/bin/gcc" if name == "gcc" else None,
    )
    assert find_compilers() == ["gcc"]


def test_keeps_gcc_before_clang(monkeypatch: pytest.MonkeyPatch) -> None:
    # both installed -> gcc should come first because of KNOWN_COMPILERS order
    monkeypatch.setattr(compilers.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert find_compilers() == ["gcc", "clang"]
