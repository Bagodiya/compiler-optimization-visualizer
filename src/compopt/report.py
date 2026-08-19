"""Asking the compiler what it did, instead of working it out from the asm.

Everything in `detectors/` reads the finished assembly and reasons backwards:
the frame pointer is gone, so the frame must have been elided. gcc doesn't
need to be guessed at like that. `-fopt-info-all` makes it say so directly —
every pass that fired, every one that wanted to fire and couldn't, each with
the source line it was looking at.

The two disagree often enough to be interesting, which is the whole point of
the report. This module only gets the text out; reading it is the next step.
"""

import subprocess
import tempfile
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
