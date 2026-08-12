"""The two things that can happen to a call: it jumps instead, or it goes away.

Tail calls and inlining share a file because they share the question — is this
callee still reached from here — and because they answer it in opposite ways.
`detect_inlining` has to rule out a tail call before it can claim a callee was
copied in, so keeping them apart would only mean importing one into the other.
"""

from compopt.annotation import Annotation
from compopt.detectors.parsing import (
    MEMORY_BRACKETS,
    UNCONDITIONAL_JUMPS,
    instruction_line_range,
    jump_target,
    label_name,
    parse_instruction,
)

# a jump can also go through a register instead of naming anything, which is
# what a switch big enough to get a jump table compiles to. AT&T marks that
# with a '*' and Intel just writes the register bare, so the sigil doesn't
# catch both and the names have to be listed. Only the 64-bit ones: what a
# jump wants is an address, and an address here is 64 bits wide.
ADDRESS_REGISTERS = frozenset(
    {
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "rbp",
        "rsp",
        "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
    }
)

# the two spellings of a call. same story as the prologue mnemonics in
# `frame.py`: only a couple of them ever turn up, so listing them reads better
# than taking letters off the end.
CALL_MNEMONICS = ("call", "callq")

TAIL_CALL = "tail call optimization"
TAIL_CALL_DESCRIPTION = (
    "the last thing the function does is hand over to another one, and it "
    "jumps there rather than calling it and waiting, so nothing comes back "
    "here — the callee borrows this frame and returns straight to our caller"
)

INLINING = "inlining"
INLINING_DESCRIPTION = (
    "a function this one used to call isn't called any more, because its body "
    "was copied in here instead — so the work happens in place, with no "
    "arguments to arrange, no jump away and no trip back"
)


def _local_labels(asm: str) -> set[str]:
    """Every label the body defines, lowercased.

    `loops.has_loop_branch` collects these as it walks, because it cares about
    which ones it has already gone past. Here the direction doesn't matter — a
    jump naming any of them is staying inside this function, and a jump naming
    something else is leaving it, which is the whole question below.
    """
    labels: set[str] = set()
    for line in asm.splitlines():
        label = label_name(line)
        if label is not None:
            labels.add(label)
    return labels


def _is_indirect(operand: str) -> bool:
    """True when a jump goes through a register or through memory.

    `jmp *%rax` and `jmp qword ptr [rax]` both land here. The address only
    exists once the program is running, so there's no name in the operand to
    compare against anything, and the table it came out of points back into
    this same function anyway.
    """
    if operand.startswith("*"):
        return True
    if any(bracket in operand for bracket in MEMORY_BRACKETS):
        return True
    return operand in ADDRESS_REGISTERS


def _final_instruction(asm: str) -> tuple[int, str, list[str]] | None:
    """The last instruction in the body: its line, mnemonic and operands.

    None when there are no instructions in there at all. Whatever the compiler
    leaves underneath doesn't count — gcc writes .cfi_endproc and a .size line
    after the last instruction and clang doesn't, and neither of them is code,
    so the function ends in the same place either way.
    """
    found = None
    for number, line in enumerate(asm.splitlines(), start=1):
        parsed = parse_instruction(line)
        if parsed is not None:
            mnemonic, operands = parsed
            found = (number, mnemonic, operands)
    return found


def tail_jump_line(asm: str) -> int | None:
    """Which line jumps away to another function, when that's how the body ends.

    Three things have to hold, and each one rules out something that looks
    similar. The last instruction is an unconditional jump: a conditional one
    falls through to more of our own code, so the function isn't finished with
    it. The target is a name rather than a register, since an indirect jump is
    a jump table landing back inside here. And the name isn't a label this body
    defines, because a jump to one of those is a loop or an if.
    """
    final = _final_instruction(asm)
    if final is None:
        return None

    number, mnemonic, operands = final
    if mnemonic not in UNCONDITIONAL_JUMPS or len(operands) != 1:
        return None

    target = operands[0]
    if _is_indirect(target) or target in _local_labels(asm):
        return None
    return number


def detect_tail_call(asm: str) -> Annotation | None:
    """Spot a call at the end of a function that became a jump.

    `return helper(x);` at -O0 is a call and then a ret. helper gets a stack
    frame of its own, does its work, returns here, and this function turns
    round and passes the same value straight back up. That middle stop is
    wasted — there is nothing left to do here once helper is done. So the
    optimizer drops the ret and jumps to helper instead of calling it: helper
    reuses the frame that's already there and returns to our caller directly.

    Worth a frame and a return each time, and it's also what lets deep
    recursion finish. A tail-recursive function that jumps instead of calling
    stops stacking frames up, so it can't run the stack out.

    One body is enough here, which the two-body detectors can't say. They need
    the -O0 build because the shape they look for could have been written that
    way by hand. This one couldn't: C has no way to say "go to that function
    and don't come back", so a jump standing where a call belongs is always
    something the compiler did.

    A function that tail-calls itself is the case this misses. That one comes
    out as a jump to a label just inside the same function, which is a loop by
    the time it's asm and reads as one — `loops.has_loop_branch` is what finds
    it.
    """
    line = tail_jump_line(asm)
    if line is None:
        return None

    return Annotation(TAIL_CALL, line, line, TAIL_CALL_DESCRIPTION)


def _call_target(mnemonic: str, operands: list[str]) -> str | None:
    """The function a call names, or None when the line isn't a direct call.

    A call through a pointer — `call *%rax`, or the Intel spelling through
    memory — drops out here, the same way `parsing.jump_target` lets an
    indirect jump fall through. There's no name in the operand to hold against
    the other build, so a callee reached that way can't be followed either
    side.
    """
    if mnemonic not in CALL_MNEMONICS or len(operands) != 1:
        return None
    target = operands[0]
    return None if _is_indirect(target) else target


def called_functions(asm: str) -> set[str]:
    """Every function the body calls by name, lowercased.

    A set and not a count, because the question underneath is whether a given
    callee is still reached at all, and two calls to the same helper answer
    that once between them. The cost is a function called from two places with
    only one of them inlined: the other call keeps the name in here and the
    half that was copied in goes unreported.
    """
    names: set[str] = set()
    for line in asm.splitlines():
        parsed = parse_instruction(line)
        if parsed is None:
            continue
        target = _call_target(*parsed)
        if target is not None:
            names.add(target)
    return names


def _outward_jumps(asm: str) -> set[str]:
    """Names the body jumps to that aren't labels it defines itself.

    Which is to say the functions it leaves for without meaning to come back —
    tail calls, by the time they're asm. `tail_jump_line` asks the same thing
    of the last instruction only, since that's where a tail call has to be;
    here every jump counts, because all that matters below is whether the name
    is still reachable somehow.
    """
    local = _local_labels(asm)
    targets: set[str] = set()
    for line in asm.splitlines():
        parsed = parse_instruction(line)
        if parsed is None:
            continue
        target = jump_target(*parsed)
        if target is not None and not _is_indirect(target) and target not in local:
            targets.add(target)
    return targets


def inlined_callees(baseline: str, optimized: str) -> set[str]:
    """Functions the baseline calls that the optimized build never reaches.

    Neither by a call nor by a jump. Taking the jumps out is what keeps a tail
    call from landing in here: `call helper` becoming `jmp helper` loses the
    call, but helper still runs, so nothing about it was copied in.
    """
    gone = called_functions(baseline) - called_functions(optimized)
    return gone - _outward_jumps(optimized)


def detect_inlining(baseline: str, optimized: str) -> Annotation | None:
    """Spot a callee whose body was copied into this function.

    A call isn't free. The arguments have to go in the registers the ABI asks
    for, the return address gets pushed, control leaves, and the callee sets up
    a frame of its own before it does any of the work. For a helper that's a
    few instructions long, all of that costs more than the helper does. So the
    compiler writes the helper's instructions straight into the caller and
    deletes the call — and then the two sides are one body, which is what makes
    it worth doing beyond the call itself: constants from here fold into work
    from there, and the optimizations after it see the whole thing at once.

    Two bodies again, for the reason `deadcode.detect_dead_code_elimination`
    needs them. Inlining leaves no shape behind — the copied instructions are
    ordinary instructions and nothing marks where they came from. All you can
    see is a name that used to be called and isn't. That has to come from the
    -O0 build.

    A call that turned into a jump is excluded, since the callee is still doing
    its own work and only the return trip went; that one is `detect_tail_call`.
    What isn't excluded is a call the compiler deleted outright, which it may
    do when it can prove the callee has no side effects and nothing reads the
    result. That reads exactly the same from here and gets reported as
    inlining. Same trade as the other detectors: the asm says the callee went,
    not where it went to.

    The annotation covers the whole optimized body rather than a line, because
    the copied instructions are spread through it with nothing separating them
    from the caller's own. Narrowing that down means asking gcc: -fopt-info-inline
    names the callee and the line it was inlined into, so nothing is guessed.
    """
    if not baseline.strip() or not optimized.strip():
        return None

    if not inlined_callees(baseline, optimized):
        return None

    span = instruction_line_range(optimized)
    if span is None:
        return None

    start, end = span
    return Annotation(INLINING, start, end, INLINING_DESCRIPTION)
