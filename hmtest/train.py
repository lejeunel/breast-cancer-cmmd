from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated


def train(
    run_path: Annotated[Path, typer.Argument()],
    assets_path: Annotated[Path, typer.Option()] = "assets",
    image_size: Annotated[int, typer.Option(help="size of input image")] = 320,
    num_train_batches: Annotated[
        int, typer.Option(help="num. of images per epoch")
    ] = 120,
    num_val_batches: Annotated[int, typer.Option(help="num. of images per epoch")] = 40,
    num_prev_batches: Annotated[
        int, typer.Option(help="num. of images to preview")
    ] = 1,
    batch_size: Annotated[int, typer.Option()] = 16,
    num_workers: Annotated[
        int,
        typer.Option(help="number of jobs for parallel batch construction"),
    ] = 8,
    learning_rate: Annotated[float, typer.Option()] = 1e-4,
    weight_decay: Annotated[float, typer.Option()] = 1e-6,
    n_epochs: Annotated[int, typer.Option()] = 200,
    cp_path: Annotated[
        Optional[Path], typer.Option(help="checkpoint to resume from")
    ] = None,
    pretrained: Annotated[Optional[bool], typer.Option()] = True,
):
    pass
