# Detected vs reported optimizations

`compopt annotate` answers "what did the optimizer do to this function?" twice,
by two routes that share almost nothing:

- **detected** — the default. Compile at `-O0` and at the level you asked for,
  read both bodies, and work backwards from the shape of the code.
  `detectors/` does this.
- **reported** — `--report`. Compile once with the compiler's own reporting
  flag on and print what it says. `passes.py` gets the answer out of whichever
  compiler is in front of it.

They disagree constantly. That isn't a bug in either of them, and this file is
about why.

## Where each one gets its facts

The detectors have the finished assembly and nothing else. Every finding is an
inference from a difference between two bodies, or from something the optimized
body doesn't have. There's no line in the asm that says "this loop was
unrolled" — there's a loop body written out four times with no jump back, and
`loops.detect_loop_unrolling` reasons from that.

The report doesn't infer anything. gcc's `-fopt-info-all` and clang's `-Rpass`
are the passes themselves printing as they run, so a record is a pass saying
what it did, or wanted to do and gave up on, at a named line of the source.

That's the whole difference, and everything below falls out of it.

## What only the detectors see

A pass has to be wired up to report before it can report, and plenty of them
aren't. Constant folding is the clearest case. `examples/const_fold.c` at `-O2`:

    $ compopt annotate examples/const_fold.c --level 2 --summary
    3 optimizations found:
      dead code elimination (lines 3-7)
      register coalescing (lines 3-7)
      constant folding (line 5)

and gcc's own account of the same compile is one record, about something else
entirely:

    const_fold.c:9:14: note: ***** Analysis failed with vector mode VOID

The arithmetic really was folded — the function returns a literal. gcc just
folded it somewhere in the middle end that has nothing to say about it. The
same goes for the frame pointer, register allocation and most of the small
cleanups: they happen, they change the asm, and no pass announces them.

So an empty report doesn't mean nothing happened.

## What only the report sees

The mirror of that: a pass that ran and left no mark the asm can be read
backwards from.

- **Missed passes.** A vectorizer that tried a loop and gave up produced no
  instructions at all, so there is nothing for a detector to find. Half of
  `loop.c`'s gcc report is this — `couldn't vectorize loop`, `not vectorized:
  unsupported data-type`, and the retries at V4SI, V8QI, V4QI on the way down.
  This is the part I find most useful and the detectors can never have it.
- **Which pass.** The report names one. A detector names the shape it saw.
- **Where in the source.** A record carries the `.c` file and line the pass was
  looking at. An annotation carries asm lines, and only reaches the source
  through `crossref.py` and the `.loc` directives.
- **Everything outside the function.** The report is per file, so `--report`
  prints records about functions `--func` didn't pick. Those come out with
  "no asm came from this line" against them rather than being dropped.

## They also disagree about the same event

`examples/loop.c` is a `for` loop summing 1..n. The detectors say:

    dead code elimination (lines 3-20)
    register coalescing (lines 3-20)

Apple clang's `-Rpass` for the same file and level says:

    loop.c:7:5  optimized (loop-delete): Loop deleted because it is invariant

and homebrew's gcc-15 says:

    loop.c:7:23  optimized: loop unrolled 1 times

Three descriptions of three genuinely different compilations. clang replaced
the loop with the closed form, which is why "dead code elimination" is the best
a detector reading only the result can do — there are fewer instructions and
they compute the answer directly, which is exactly what dead code elimination
looks like from the outside. gcc kept the loop and unrolled it once. Neither
detector finding is wrong about the asm in front of it; they're just not the
words the compiler would use, because the detector never saw the loop get
deleted, only that it was gone.

Two compilers reporting different things about the same source is normal and
worth seeing. It's why `PassReport` keeps which flag answered and the heading
prints it.

## Which one to believe

When they disagree about something the report actually covers, the report is
right — it's a pass's own account and the detector is a guess from the
leftovers. When the report is silent, that's not evidence against a detector
finding; most of what the detectors name is never reported by anybody.

The detector docstrings each say which way they err. The two that err hardest
are `folding.detect_constant_folding`, which can't tell a folded calculation
from a constant that was written as one, and
`registers.detect_register_coalescing`, which can't tell a spill that was
removed from a function that never had one. Both fire on `const_fold.c` above,
and for that file both happen to be right.

## One thing that trips everyone up on macOS

`gcc` here is Apple clang under gcc's name, so `--report` goes down the
`-Rpass` route and the heading reads:

    gcc (-Rpass) at -O2 reported 16 things:

That's the heading doing its job. Nothing picks a route by the compiler's name
— `collect_report` asks with gcc's flag, and if the compiler turns it down it
asks with clang's, because a rejected flag is the one answer that can't be a
guess. If you want a real gcc report on this machine you need homebrew's
`gcc-15`, and `--compiler` only knows the bare names `gcc` and `clang`, so for
now that means calling `collect_report` directly the way `tests/test_report.py`
does.
