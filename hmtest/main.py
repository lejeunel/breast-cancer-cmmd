import typer
from typing_extensions import Annotated
from pathlib import Path
from typing import Optional

from hmtest import dataset
from hmtest.train import train

app = typer.Typer()
app.add_typer(dataset.app, name="dset")
app.command(help="Train")(train)


if __name__ == "__main__":
    app()
