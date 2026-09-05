"""Tests for compiler detection, level spelling, and driving gcc/clang."""

from pathlib import Path

import pytest
import typer

from compopt import compilers
from compopt.compilers import (
    CompileError,
    check_level,
    compile_at_levels,
    compile_to_asm,
    find_compilers,
    install_hint,
    no_compiler_lines,
    normalize_level,
    pick_compiler,
    report_bad_request,
    suffixed_compilers,
)


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


def test_compile_to_asm_cleans_up_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    available = find_compilers()
    if not available:
        pytest.skip("no gcc/clang on this machine to test against")

    # send all temp files into our own empty dir so we can check it after
    tmpdir = tmp_path / "scratch"
    tmpdir.mkdir()
    monkeypatch.setattr(compilers.tempfile, "tempdir", str(tmpdir))

    src = tmp_path / "add.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")
    compile_to_asm(src, "1", available[0])

    # the temp dir we compiled into should be gone again
    assert list(tmpdir.iterdir()) == []


def test_compile_to_asm_bad_source_raises(tmp_path: Path) -> None:
    available = find_compilers()
    if not available:
        pytest.skip("no gcc/clang on this machine to test against")

    src = tmp_path / "broken.c"
    src.write_text("int main(void) { this is not c }\n")

    with pytest.raises(CompileError) as excinfo:
        compile_to_asm(src, "0", available[0])

    # the wrapped error should remember which compiler failed and carry
    # the actual diagnostic text, not be empty
    assert excinfo.value.compiler == available[0]
    assert excinfo.value.message


def test_compile_at_levels_defaults(tmp_path: Path) -> None:
    available = find_compilers()
    if not available:
        pytest.skip("no gcc/clang on this machine to test against")

    src = tmp_path / "add.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    results = compile_at_levels(src, available[0])

    # one entry per default level, each holding real-looking asm
    assert set(results) == {"0", "1", "2", "3"}
    for asm in results.values():
        assert "add" in asm
        assert "ret" in asm.lower()


def test_compile_at_levels_custom_set(tmp_path: Path) -> None:
    available = find_compilers()
    if not available:
        pytest.skip("no gcc/clang on this machine to test against")

    src = tmp_path / "add.c"
    src.write_text("int add(int a, int b) { return a + b; }\n")

    results = compile_at_levels(src, available[0], levels=["0", "2"])

    assert set(results) == {"0", "2"}


def test_compile_at_levels_propagates_error(tmp_path: Path) -> None:
    available = find_compilers()
    if not available:
        pytest.skip("no gcc/clang on this machine to test against")

    src = tmp_path / "broken.c"
    src.write_text("int main(void) { this is not c }\n")

    # one bad level should surface as an error, not a partial dict
    with pytest.raises(CompileError):
        compile_at_levels(src, available[0])


# these poke pick_compiler directly so they don't need a real toolchain

def testpick_compiler_flag_beats_cc(monkeypatch: pytest.MonkeyPatch) -> None:
    # an explicit --compiler should ignore whatever $CC says
    monkeypatch.setenv("CC", "clang")
    assert pick_compiler("gcc", ["gcc", "clang"]) == "gcc"


def testpick_compiler_uses_cc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CC", "clang")
    assert pick_compiler(None, ["gcc", "clang"]) == "clang"


def testpick_compiler_cc_can_be_a_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # CC is often a full path, so we match on the file name
    monkeypatch.setenv("CC", "/usr/local/bin/clang")
    assert pick_compiler(None, ["gcc", "clang"]) == "clang"


def testpick_compiler_ignores_unusable_cc(monkeypatch: pytest.MonkeyPatch) -> None:
    # CC=cc isn't something we know how to drive, so fall back to gcc-first
    monkeypatch.setenv("CC", "cc")
    assert pick_compiler(None, ["gcc", "clang"]) == "gcc"


def testpick_compiler_default_when_no_cc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CC", raising=False)
    assert pick_compiler(None, ["gcc", "clang"]) == "gcc"


def testpick_compiler_bad_flag_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CC", raising=False)
    with pytest.raises(typer.Exit):
        pick_compiler("notacc", ["gcc", "clang"])


# what we say when there's nothing to compile with


def fake_path(monkeypatch: pytest.MonkeyPatch, contents: dict[str, list[str]]) -> None:
    """Pretend PATH is exactly these directories holding exactly these names."""
    monkeypatch.setenv("PATH", ":".join(contents))
    monkeypatch.setattr(compilers.os, "listdir", lambda entry: contents[entry])


def test_suffixed_finds_a_versioned_gcc(monkeypatch: pytest.MonkeyPatch) -> None:
    # the homebrew case: a real compiler is there, just not called gcc
    fake_path(monkeypatch, {"/opt/homebrew/bin": ["gcc-15", "brew"]})
    assert suffixed_compilers() == ["gcc-15"]


def test_suffixed_skips_the_lookalikes(monkeypatch: pytest.MonkeyPatch) -> None:
    # all of these start with a compiler name and none of them compiles C
    fake_path(
        monkeypatch,
        {"/usr/bin": ["gcc-ar", "gcc-ranlib", "clang++", "clang-format", "clangd"]},
    )
    assert suffixed_compilers() == []


def test_suffixed_sorts_and_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    # the same name in two PATH entries is still one compiler
    fake_path(
        monkeypatch,
        {"/usr/local/bin": ["clang-17", "gcc-15"], "/opt/bin": ["gcc-15"]},
    )
    assert suffixed_compilers() == ["clang-17", "gcc-15"]


def test_suffixed_survives_a_bad_path_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    # PATH picks up directories that don't exist; that can't be fatal here,
    # because we're already on the way to reporting a different problem
    monkeypatch.setenv("PATH", "/nowhere/at/all")
    assert suffixed_compilers() == []


def test_no_compiler_lines_name_what_was_looked_for() -> None:
    first = no_compiler_lines()[0]
    for name in compilers.KNOWN_COMPILERS:
        assert name in first


def test_no_compiler_lines_point_at_a_suffixed_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compilers, "suffixed_compilers", lambda: ["gcc-15"])
    text = " ".join(no_compiler_lines())

    # telling someone to install gcc when gcc-15 is sitting right there is
    # the message this step existed to get rid of
    assert "gcc-15" in text
    assert "install" not in text


def test_no_compiler_lines_fall_back_to_the_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compilers, "suffixed_compilers", list)
    assert install_hint() in " ".join(no_compiler_lines())


def test_install_hint_knows_this_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compilers.platform, "system", lambda: "Darwin")
    assert "xcode-select" in install_hint()

    monkeypatch.setattr(compilers.platform, "system", lambda: "Linux")
    assert "package manager" in install_hint()

    monkeypatch.setattr(compilers.platform, "system", lambda: "Windows")
    assert install_hint()


def test_unknown_compiler_is_not_an_install_problem(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(typer.Exit):
        report_bad_request("tcc", ["gcc", "clang"])

    err = capsys.readouterr().err
    # no amount of installing makes --compiler tcc work, so don't suggest it
    assert "install" not in err
    assert "gcc, clang" in err


def test_known_compiler_missing_says_how_to_get_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(typer.Exit):
        report_bad_request("gcc", ["clang"])

    err = capsys.readouterr().err
    assert "clang" in err
    assert install_hint() in err


# level spelling: everything inside works in bare digits, but nobody types them


def test_normalize_level_leaves_a_bare_digit_alone() -> None:
    assert normalize_level("0") == "0"
    assert normalize_level("3") == "3"


def test_normalize_level_takes_the_o_off() -> None:
    # the spelling you'd copy out of a Makefile or a compiler manual
    assert normalize_level("O2") == "2"
    assert normalize_level("-O2") == "2"
    assert normalize_level("o2") == "2"


def test_normalize_level_passes_nonsense_through() -> None:
    # check_level has to be the one to reject it, and it can only do that if
    # the string still looks like what the user typed
    assert normalize_level("fast") == "fast"
    assert normalize_level("9") == "9"


def test_normalize_level_never_returns_empty() -> None:
    # "O" on its own would normalize down to nothing, which would then be
    # rejected with a blank name in the message
    assert normalize_level("O") == "O"
    assert normalize_level("-") == "-"


def test_check_level_accepts_every_level_we_compile() -> None:
    for level in compilers.DEFAULT_LEVELS:
        check_level("--level", level)  # no exception


def test_check_level_rejects_one_we_dont() -> None:
    with pytest.raises(typer.Exit):
        check_level("--from", "9")


def test_check_level_rejects_a_word() -> None:
    with pytest.raises(typer.Exit):
        check_level("--to", "fast")
