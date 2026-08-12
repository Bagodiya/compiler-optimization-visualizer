# Architecture

Notes on how compopt is put together, mostly for my own sake so I don't have to
re-read every module when I come back to this after a break. Nothing here is set
in stone — it just describes the code as it stands.

## The big picture

Three commands, and they all open the same way: compile the file at some `-O`
levels, throw away the assembler bookkeeping, pull out one function. What they
do with that function afterwards is the only part that differs.

```
source.c
   │  compile at the levels we need  (compilers.py)
   ▼
{ "0": asm, "2": asm, ... }
   │  drop the noisy directives      (asm.py: strip_directives)
   ▼
   │  keep just one function          (asm.py: isolate_function)
   ▼
one body per level
   ├─ show      lay them out in columns          (render.py)
   ├─ diff      line up two of them              (diff.py)
   └─ annotate  run the detectors over -O0 + one (detectors/)
   ▼
printed to the terminal
```

`show` compiles all four levels because it's showing you all four. `diff` and
`annotate` compile only the two they need — the others would be work nobody
asked to see.

The CLI layer sits on top and doesn't do any of the real work itself — it
parses arguments, hands off to `run_show`/`run_diff`/`run_annotate`, and turns
a `CompileError` into a clean message.

## Modules

### `cli.py`
The Typer app. Defines `--version` and the three commands, reads their options
and calls the matching `run_*`. The one bit of logic it keeps is catching
`CompileError` so a broken source file prints a clean message instead of a
traceback.

### `show.py`
The `show` command's actual body. This is the orchestrator: it checks the path
exists, figures out which compiler to run, kicks off the compile, works out how
many columns fit, and pulls the wanted function out of each level's assembly
before handing the columns to the renderer. `_function_body` is the little
strip-then-isolate combo applied to a single level.

### `compilers.py`
Everything that talks to gcc/clang, plus the two questions every command has to
answer before it can compile anything: which compiler, and which levels.

`find_compilers` uses `shutil.which` so we only ever report a compiler we can
really run. `pick_compiler` sorts out the gcc-vs-clang question — an explicit
`--compiler` wins but has to be installed, otherwise we look at `$CC` the way
make does, and if nothing's chosen we fall back to whatever came first.
`choose_compiler` is those two together, which is what the commands call.
`normalize_level` takes `2`, `O2` or `-O2` and gives back the bare digit
everything inside works in, and `check_level` rejects anything that isn't one
we can compile before we shell out to a compiler that would reject it anyway.

`compile_to_asm` runs one `-O` level in a throwaway temp dir and reads the `.s`
file back — it doesn't use `check=True` on purpose, because we want to grab
stderr and wrap it in our own `CompileError` rather than let a raw
`CalledProcessError` escape.

`compile_at_levels` runs the levels it's given, defaulting to all four. Each is
a separate compiler process and they spend most of their time just waiting, so
they get fanned out across a `ThreadPoolExecutor` instead of run one after
another. If any level fails the error propagates — a half-finished comparison
isn't worth showing.

### `asm.py`
The text cleanup, and the part with the most fiddly edge cases. Compiler output
is full of bookkeeping that says nothing about optimization, so
`strip_directives` throws away the noise (`.cfi_*`, `.file`, `.section` and
friends) while keeping instructions and the local `.L` labels.

`isolate_function` grabs a single function — everything from its label down to
the next function label. The tricky bit is telling a real function label apart
from the noise, which `_label_name` handles: it skips indented lines
(instructions), skips comment-only lines, and strips a trailing comment before
checking for the colon. That's mostly there to cope with macOS clang, which
prefixes names with an underscore (`_add:`), tacks `## @add` onto the label
line, and emits comment lines like `## %bb.0:`. `_matches` deals with the same
underscore prefix so `--func add` still finds `_add`.

The one that bit hardest is `_is_macho`. On ELF a local label starts with a dot
(`.L2:`) and the dot test catches it, but Mach-O local labels are a bare `L` at
column 0 — exactly where a function label goes. clang writes `LBB0_1:` for a
basic block, GNU gcc targeting Darwin wraps every function in `LFB0:`/`LFE0:`,
and both scatter things like `EH_frame1:` around. Counted as functions those do
real damage, because `isolate_function` stops at the next function label: a
function was being cut off at its own first local label with everything below
it silently dropped, which `show` had been doing quietly for a while.

Chasing that with a list of prefixes to skip is a losing game — the first
version of the fix knew clang's four and still broke the moment a real GNU gcc
ran. The rule that actually holds is that Mach-O gives every C symbol a leading
underscore, so there a function label is one starting with `_` and everything
else at column 0 is the assembler talking to itself. Which format we're looking
at is decided once per body, from underscored symbols being present and dotted
local labels being absent — both halves, so that one `_start` in a Linux build
can't flip the rule over and hide every function that isn't underscored.

### `render.py`
Turns cleaned assembly into what you see on screen. `levels_for_width` decides
whether the terminal is wide enough for all four columns or has to drop back to
just `-O0` vs `-O2` — cram four columns into a narrow terminal and every line
folds into soup. `render_columns` builds a rich `Table`, one row, with a
right-aligned line-number gutter on the left so rows are easy to point at. Long
lines are cut with an ellipsis instead of wrapping so each instruction stays on
its own row, lined up with its number.

`highlight_asm` colors each line: bold for labels, one color for the mnemonic,
another for registers and immediates. With `--no-color` it skips all of that and
hands back plain text, which is what you want when the output is being piped.

`render_annotated` is the same shape with a notes column instead of a second
level of asm, and `annotation_gutter` fills that column — one entry per asm
line, the name against the line it starts on and a marker down the rest of the
span. Nothing in that column wraps, and that's load-bearing: it's one cell of
text lined up against another, so a note folding onto a second row would push
every note below it down one and they'd all point at the wrong instruction.

One thing worth remembering: assembly comes out tab-indented, and rich measures
a tab as one cell but draws it as eight. That makes short lines look too wide and
get chopped for no reason, so the tabs get expanded to spaces before rendering.

### `diff.py`
`diff_lines` leans on `difflib.SequenceMatcher` to tag each line equal, added or
removed; `trim_context` folds the long runs of unchanged lines into a "gap"
marker; `render_diff` and `highlight_diff` put the `+`/`-` gutter on. There's
also `unified_diff` for the portable `diff -u` form, which lets difflib do the
`@@` line-number arithmetic rather than working it out by hand.

Two things in `run_diff` are easy to get backwards. `missing_message` runs
first, because a function that isn't in the optimized build at all would
otherwise diff as every line being deleted one at a time, which reads like the
optimizer took the code apart instead of inlining it. And `is_identical` gets
asked before trimming, not after — trimming turns a run of unchanged lines into
a gap entry, so a body that didn't change at all stops looking like one.

### `annotation.py`
Just the `Annotation` type. It's frozen and validates its own line range on the
way in, since a bad range turns into a note pointing at the wrong instruction,
which is miserable to debug from the output alone. It sits on its own because
the renderer wants it as much as the detectors do, and it doesn't need to know
anything about either.

### `detectors/`
One optimization per module — `frame.py`, `folding.py`, `registers.py`,
`deadcode.py`, `loops.py`, `calls.py`, `arithmetic.py`, `branches.py`,
`vectors.py` — each holding the detector, whatever helpers only it uses, and
its own name and description string. `calls.py` is the one with two in it: tail
calls and inlining both turn on whether a callee is still reached from here,
and `detect_inlining` has to rule out a tail call before it can claim anything,
so splitting them would only mean one importing the other.

`parsing.py` underneath is the shared part: `parse_instruction` turns a line
into a mnemonic and operands with the AT&T `%` off and everything lowercased,
which is what lets one detector cope with both syntaxes, plus the register
names and jump mnemonics more than one of them needs. Anything only one
detector asks for stays in that detector's file.

`__init__.py` collects them. The two tuples split the detectors by what they
need: some can work off the optimized body on its own, because what they look
for is visible in it — a prologue that isn't there, a literal where a
calculation used to be. The rest need `-O0` as well, because what they look for
is something that stopped being there, and absence has no shape of its own.
`find_annotations` runs both groups and sorts what comes back, and
`DESCRIPTIONS` is the same set keyed by name for `--explain`.

They report the shape they see rather than what the compiler says it did, so
several of them can't separate an optimization from source that was written
that way to begin with. Each docstring says which way it errs. Reading gcc's
`-fopt-info` would settle those, and that's the obvious next step.

### `annotate.py`
What's left of the command once the detecting moved out: check the path, pick
the compiler, compile `-O0` and the level asked for, pull the wanted function
out of each, and print the asm with the findings beside it. `--explain` short-
circuits all of that, which is why the path argument is optional.

## A couple of decisions

- **Levels keyed by the bare digit.** Everything passes levels around as `"0"`,
  `"1"`, `"2"`, `"3"` and only sticks the `-O` on at display time. Keeps the
  dict keys and the compiler flag from drifting apart.
- **Temp dirs, not files in the tree.** Compiling writes to a
  `TemporaryDirectory` that's cleaned up as soon as we've read the `.s` back, so
  nothing gets left behind in the project.
- **The CLI stays thin.** All the real logic sits in `show`/`compilers`/`asm`/
  `render`, which are plain functions that are easy to test without going
  through Typer. `cli.py` is basically wiring.

## Where this is going

The detectors are the weak point, and they're weak in a known way: they read
the assembly and infer, which means a handful of them can't tell an
optimization from source that happened to be written that way. gcc will just
tell you — `-fopt-info-inline` names the callee and the line it was inlined
into, `-fopt-info-loop` says how many copies an unrolled body got. Parsing that
and cross-referencing it against what the detectors found would turn most of
the guesses into facts, and would catch the cases they miss outright (partial
unrolling, a callee inlined at one call site but not another).

After that, a `report` command that runs the whole thing over a file and
summarises per function, rather than one function at a time.
