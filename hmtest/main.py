import typer
from typing_extensions import Annotated
from pathlib import Path
from typing import Optional

from hmtest.dataset import fetch_raw_data
from hmtest.train import train

app = typer.Typer()
app.command(help="Fetch raw data")(fetch_raw_data)
app.command(help="Train")(train)


if __name__ == "__main__":
    app()
