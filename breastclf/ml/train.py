import datetime
from pathlib import Path
from typing import Optional

import typer
from breastclf.ml.callbacks import make_callbacks
from breastclf.ml.dataloader import make_dataloaders
from breastclf.ml.utils import save_to_yaml
from breastclf.ml.model import BreastClassifier
from typing_extensions import Annotated
from breastclf.ml.trainer import Trainer

from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib


def train(
    meta_path: Annotated[Path, typer.Argument(help="path to meta-data csv file")],
    image_root_path: Annotated[Path, typer.Argument(help="root path to image files")],
    out_root_path: Annotated[Path, typer.Argument(help="output path", exists=True)],
    exp_name: Annotated[
        str, typer.Argument(help="name of experiment (used for logging)")
    ],
    append_datetime_to_exp: Annotated[
        Optional[bool],
        typer.Option(help="Whether we add a datetime stamp to run directory"),
    ] = True,
    image_size: Annotated[int, typer.Option(help="size of input image")] = 1024,
    batch_size: Annotated[int, typer.Option()] = 16,
    n_batches_per_epoch: Annotated[int, typer.Option()] = 50,
    n_workers: Annotated[
        int, typer.Option(help="Num of parallel processes in data loader")
    ] = 8,
    learning_rate: Annotated[float, typer.Option()] = 5e-5,
    weight_decay: Annotated[float, typer.Option()] = 0,
    checkpoint_period: Annotated[int, typer.Option()] = 1,
    n_epochs: Annotated[int, typer.Option()] = 20,
    seed: Annotated[int, typer.Option()] = 0,
    cuda: Annotated[Optional[bool], typer.Option(help="use GPU")] = True,
    resume_cp_path: Annotated[
        Optional[Path], typer.Option(help="checkpoint to resume from")
    ] = None,
    fusion: Annotated[
        str,
        typer.Option(
            help="breast-wise fusion mode: ['max-feats', 'mean-feats', 'output', 'concat-feats']"
        ),
    ] = "output",
    lftype: Annotated[float, typer.Option(help="loss factor for type")] = 1.0,
    lfabnorm: Annotated[float, typer.Option(help="loss factor for abnormality")] = 0.0,
):
    """
    Training routine.
    """

    torch.manual_seed(seed)

    if append_datetime_to_exp:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_path = out_root_path / (current_time + "_" + exp_name)
    else:
        run_path = out_root_path / exp_name

    run_path.mkdir()

    save_to_yaml(run_path / "cfg.yaml", locals())

    matplotlib.use("Agg")

    dataloaders = make_dataloaders(
        image_root_path,
        meta_path,
        meta_backup_path=run_path,
        image_size=image_size,
        batch_size=batch_size,
        n_batches_per_epoch=n_batches_per_epoch,
        n_workers=n_workers,
    )

    device = torch.device("cuda") if cuda else torch.device("cpu")
    model = BreastClassifier(fusion_mode=fusion).to(device)

    optim = Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    curr_epoch = 0
    past_metrics = []
    if resume_cp_path is not None:
        print(f"loading weights from {resume_cp_path}")
        dict_ = torch.load(resume_cp_path)
        model.load_state_dict(dict_["model_state_dict"])
        optim.load_state_dict(dict_["optimizer_state_dict"])
        curr_epoch = dict_["epoch"]
        past_metrics = dict_["metrics"]

    trainer = Trainer(
        model,
        optim,
        device=device,
        start_epoch=curr_epoch,
        loss_factors={"type": lftype, "abnorm": lfabnorm},
    )

    tboard_train_writer = SummaryWriter(str(run_path / "log" / "train"))
    tboard_val_writer = SummaryWriter(str(run_path / "log" / "val"))

    train_clbks = make_callbacks(
        tboard_writer=tboard_train_writer,
        mode="train",
        binarizer_type=dataloaders["train"].dataset.binarizer_type,
    )
    val_clbks = make_callbacks(
        tboard_writer=tboard_val_writer,
        model=model,
        optimizer=optim,
        mode="val",
        binarizer_type=dataloaders["train"].dataset.binarizer_type,
        checkpoint_root_path=run_path / "checkpoints",
        checkpoint_period=checkpoint_period,
        past_metrics=past_metrics,
    )

    for e in range(curr_epoch, n_epochs):
        print(f"Epoch {e+1}/{n_epochs}")
        trainer.train_one_epoch(dataloaders["train"], callbacks=train_clbks)
        trainer.eval(dataloaders["val"], callbacks=val_clbks)
