import typer
from breastclf.ml.split import split
from breastclf.ml.train import train
from breastclf.ml.test import test

app = typer.Typer()
app.command(help="Split into train, validation and testing set")(split)
app.command(help="Train and validate model")(train)
app.command(help="Test model")(test)
