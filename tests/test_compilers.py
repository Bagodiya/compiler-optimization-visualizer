"""Tests for compiler detection."""

import subprocess
from pathlib import Path

import pytest

from compopt import compilers
from compopt.compilers import compile_to_asm, find_compilers


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


def test_compile_to_asm_real(tmp_path: Path) -> None:
    available = find_compilers()
    if not available:
        pytest.skip("no gcc/clang on this machine to test against")

    src = tmp_path / "add.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    asm = compile_to_asm(src, "2", available[0])

    # should look like real assembly and mention the function we compiled
    assert "add" in asm
    assert "ret" in asm.lower()


def test_compile_to_asm_bad_source_raises(tmp_path: Path) -> None:
    available = find_compilers()
    if not available:
        pytest.skip("no gcc/clang on this machine to test against")

    src = tmp_path / "broken.c"
    src.write_text("int main(void) { this is not c }\n")

    with pytest.raises(subprocess.CalledProcessError):
        compile_to_asm(src, "0", available[0])
