# Architecture

Notes on how compopt is put together, mostly for my own sake so I don't have to
re-read every module when I come back to this after a break. Nothing here is set
in stone — it just describes the code as it stands.

## The big picture

Right now there's really one command that does work: `compopt show <file.c>`.
It compiles the file at a few `-O` levels and prints the assembly for one
function side by side so you can eyeball what changed between levels.

Everything hangs off a small pipeline. The source file goes in one end and a
rich table comes out the other:

```
source.c
   │  compile at -O0/-O1/-O2/-O3   (compilers.py)
   ▼
{ "0": asm, "1": asm, ... }
   │  drop the noisy directives     (asm.py: strip_directives)
   ▼
   │  keep just one function         (asm.py: isolate_function)
   ▼
[ ("-O0", body), ("-O2", body), ... ]
   │  colorize + lay out in columns  (render.py)
   ▼
printed to the terminal
```

The CLI layer sits on top and doesn't do any of the real work itself — it just
parses arguments and hands off to `show.run_show`.

## Modules

### `cli.py`
The Typer app. Defines the `--version` flag and the `show` command, reads the
options (`--func`, `--no-color`, `--width`, `--compiler`) and calls
`run_show`. The one bit of logic it keeps is catching `CompileError` so a
broken source file prints a clean message instead of a traceback.

### `show.py`
The `show` command's actual body. This is the orchestrator: it checks the path
exists, figures out which compiler to run, kicks off the compile, works out how
many columns fit, and pulls the wanted function out of each level's assembly
before handing the columns to the renderer.

Two helpers live here. `_pick_compiler` sorts out the gcc-vs-clang question: an
explicit `--compiler` wins but has to be installed, otherwise we look at `$CC`
the way make does, and if nothing's chosen we fall back to whatever
`find_compilers` found first. `_function_body` is just the little
strip-then-isolate combo applied to a single level.

### `compilers.py`
Everything that talks to gcc/clang. `find_compilers` uses `shutil.which` so we
only ever report a compiler we can really run. `compile_to_asm` runs one
`-O` level in a throwaway temp dir and reads the `.s` file back — it doesn't use
`check=True` on purpose, because we want to grab stderr and wrap it in our own
`CompileError` rather than let a raw `CalledProcessError` escape.

`compile_at_levels` runs all four levels. Each level is a separate compiler
process and they spend most of their time just waiting, so they get fanned out
across a `ThreadPoolExecutor` instead of run one after another. If any level
fails the error propagates — a half-finished comparison isn't worth showing.

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

One thing worth remembering: assembly comes out tab-indented, and rich measures
a tab as one cell but draws it as eight. That makes short lines look too wide and
get chopped for no reason, so the tabs get expanded to spaces before rendering.

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

The plan has diff, annotate, and report commands coming after this. The shape
should carry over: `compile_at_levels` already gives every level's assembly, and
`asm.py` already isolates a function, so diffing two levels or scanning one for
optimization patterns is mostly new logic on top of what's here rather than a
rewrite.
