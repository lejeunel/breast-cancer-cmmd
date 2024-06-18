import typer
from typing_extensions import Annotated
from pathlib import Path
from typing import Optional

from breastclf import cmmd
from breastclf import ml

app = typer.Typer()
app.add_typer(
    cmmd.app,
    name="cmmd",
    help="Tools to build the Chinese Mammography Database (CMMD) dataset",
)
app.add_typer(
    ml.app,
    name="ml",
    help="Model training, validation, ...",
)


if __name__ == "__main__":
    app()
