# compiler-optimization-visualizer

A command-line tool for inspecting and comparing compiler optimization output.

Compiles a C source file at multiple optimization levels and shows what each
optimization pass actually changed in the generated assembly. Useful for
learning what `-O1`, `-O2`, and `-O3` really do, debugging performance
differences, and understanding compiler behavior on real code.

## Requirements

- Python 3.11 or newer
- A C compiler available on PATH (`gcc` or `clang`)

## Installation

```sh
git clone https://github.com/Bagodiya/compiler-optimization-visualizer.git
cd compiler-optimization-visualizer
pip install -e .
```

## Usage

Show optimization levels side by side. On a wide terminal that's all four,
otherwise `-O0` against `-O2`:

```sh
compopt show examples/loop.c
```

Diff two specific levels. `-C` trims the unchanged lines around each change,
and `-u` gives you a normal unified diff you can paste somewhere:

```sh
compopt diff examples/loop.c --from O0 --to O3
compopt diff examples/loop.c -C 1 -u
```

Name the optimizations the compiler applied. `--summary` drops the assembly and
keeps just the findings:

```sh
compopt annotate examples/const_fold.c --level O2
compopt annotate examples/loop.c --summary
```

`--explain` looks one up by name, with or without a file to look at:

```sh
compopt annotate --explain register-coalescing
```

Levels can be written `2`, `O2` or `-O2` — they all mean the same thing. Pass
`--func` to any of the three to pick a function when the file has more than
one, and `--compiler gcc` or `--compiler clang` to force a toolchain (`$CC` is
honoured otherwise).

## What annotate can and can't tell you

The detectors work off the shape of the assembly, comparing the optimized
build against `-O0`. That's enough to recognise ten things:

| | |
|---|---|
| stack frame elimination | dead code elimination |
| constant folding | loop unrolling |
| register coalescing | tail call optimization |
| strength reduction | inlining |
| branch elimination | vectorization |

It does mean some of them are inferences rather than facts. A function written
as `return 200;` is indistinguishable from one whose arithmetic got folded down
to 200, and a function that never needed a stack slot looks exactly like one
whose spills were optimized away. Each detector's docstring says which way it
errs.

Reading gcc's own `-fopt-info` output would settle those cases, and that's
where this is going next.

## Project Structure

```
compiler-optimization-visualizer/
├── src/compopt/    Package source
├── tests/          Unit tests
├── examples/       Sample C programs
├── docs/           Documentation
└── pyproject.toml  Package metadata
```

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
