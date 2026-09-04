"""Tests for the --info output."""

import pytest

from compopt import compilers, info
from compopt.compilers import compiler_version
from compopt.info import compiler_lines


def test_version_line_is_one_line() -> None:
    available = info.find_compilers()
    if not available:
        pytest.skip("no gcc/clang on this machine to ask")

    banner = compiler_version(available[0])

    assert "\n" not in banner
    assert banner != "unknown"


def test_version_of_something_that_is_not_a_compiler() -> None:
    # a name that isn't on PATH raises inside subprocess, and --info still has
    # to print something for it
    assert compiler_version("definitely-not-a-real-compiler") == "unknown"


def test_version_when_the_compiler_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    class Failed:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(compilers.subprocess, "run", lambda *args, **kwargs: Failed())
    assert compiler_version("gcc") == "unknown"


def test_one_line_per_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(info, "find_compilers", lambda: ["gcc", "clang"])
    monkeypatch.setattr(info.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(info, "compiler_version", lambda name: f"{name} 1.2.3")

    lines = compiler_lines()

    assert len(lines) == 2
    assert "gcc" in lines[0]
    assert "/usr/bin/gcc" in lines[0]
    assert "gcc 1.2.3" in lines[0]


def test_says_so_when_nothing_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(info, "find_compilers", list)
    assert compiler_lines() == ["  none found on PATH"]


def test_path_falls_back_when_which_comes_back_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    # find_compilers said it was there and which now says it isn't, which
    # shouldn't be possible but shouldn't crash --info either
    monkeypatch.setattr(info, "find_compilers", lambda: ["gcc"])
    monkeypatch.setattr(info.shutil, "which", lambda _name: None)
    monkeypatch.setattr(info, "compiler_version", lambda _name: "gcc 1.2.3")

    assert "?" in compiler_lines()[0]
