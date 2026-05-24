"""Command-line interface for compopt."""

import typer

app = typer.Typer(
    name="compopt",
    help="Inspect and compare compiler optimization output.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the compopt version."""
    typer.echo("compopt 0.0.1")


if __name__ == "__main__":
    app()
