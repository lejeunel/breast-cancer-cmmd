import typer
from hmtest.split import split

app = typer.Typer()
app.command(help="Split into train, validation and testing set")(split)
