import typer
from typing_extensions import Annotated
from pathlib import Path
from typing import Optional

from hmtest import cmmd
from hmtest import mdl

app = typer.Typer()
app.add_typer(
    cmmd.app,
    name="cmmd",
    help="Tools to build the Chinese Mammography Database (CMMD) dataset",
)
app.add_typer(mdl.app, name="mdl", help="Model training and testing")


if __name__ == "__main__":
    app()
