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

Show optimization levels side by side:

```sh
compopt show examples/loop.c
```

Diff two specific levels:

```sh
compopt diff examples/loop.c --from O0 --to O3
```

Annotate detected optimizations:

```sh
compopt annotate examples/loop.c --level O2
```

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
