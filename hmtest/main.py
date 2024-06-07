import typer
from typing_extensions import Annotated
from pathlib import Path
from typing import Optional

app = typer.Typer()


@app.command(help="Train model")
def train(
    out: Annotated[Path, typer.Argument(help="path to output unified csv file")],
    base_path: Annotated[
        Optional[Path], typer.Option(help="path where csv files are located")
    ] = "assets",
):
    pass


if __name__ == "__main__":
    app()
