import datetime
from pathlib import Path
from typing import Optional

import typer
from hmtest.ml.callbacks import make_callbacks
from hmtest.ml.dataloader import make_dataloaders
from hmtest.ml.utils import save_to_yaml
from hmtest.ml.model import BreastClassifier
from typing_extensions import Annotated
from hmtest.ml.trainer import Trainer
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torch


def train(
    meta_path: Annotated[Path, typer.Argument(help="path to meta-data csv file")],
    image_root_path: Annotated[Path, typer.Argument(help="root path to image files")],
    out_root_path: Annotated[Path, typer.Argument(help="output path", exists=True)],
    exp_name: Annotated[
        str, typer.Argument(help="name of experiment (used for logging)")
    ],
    image_size: Annotated[int, typer.Option(help="size of input image")] = 512,
    batch_size: Annotated[int, typer.Option()] = 16,
    n_workers: Annotated[
        int, typer.Option(help="Num of parallel processes in data loader")
    ] = 4,
    learning_rate: Annotated[float, typer.Option()] = 1e-5,
    weight_decay: Annotated[float, typer.Option()] = 0,
    checkpoint_period: Annotated[int, typer.Option()] = 2,
    n_epochs: Annotated[int, typer.Option()] = 50,
    n_batches_per_epoch: Annotated[int, typer.Option()] = 40,
    seed: Annotated[int, typer.Option()] = 42,
    resume_cp_path: Annotated[
        Optional[Path], typer.Option(help="checkpoint to resume from")
    ] = None,
):
    """
    Training routine.
    """

    torch.manual_seed(seed)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    run_path = out_root_path / (current_time + "-" + exp_name)
    run_path.mkdir(exist_ok=True)

    save_to_yaml(run_path / "cfg.yaml", locals())

    dataloaders = make_dataloaders(
        image_root_path,
        meta_path,
        image_size=image_size,
        batch_size=batch_size,
        n_batches_per_epoch=n_batches_per_epoch,
        n_workers=n_workers,
    )

    model = BreastClassifier()

    optim = Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = BCEWithLogitsLoss(reduction="none")

    trainer = Trainer(model, optim, criterion)

    tboard_train_writer = SummaryWriter(str(run_path / "log" / "train"))
    tboard_val_writer = SummaryWriter(str(run_path / "log" / "val"))
    train_clbks = make_callbacks(
        tboard_train_writer,
        model=model,
        optimizer=optim,
        mode="train",
        checkpoint_root_path=run_path / "checkpoints",
        checkpoint_period=checkpoint_period,
    )
    val_clbks = make_callbacks(tboard_val_writer, mode="val")

    for e in range(n_epochs):
        print(f"Epoch {e+1}/{n_epochs}")
        trainer.train_one_epoch(dataloaders["train"], callbacks=train_clbks)
        trainer.eval(dataloaders["val"], callbacks=val_clbks)
