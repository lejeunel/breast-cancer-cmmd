import datetime
from pathlib import Path
from typing import Optional

import typer
from hmtest.ml.callbacks import make_callbacks
from hmtest.ml.dataloader import make_dataloaders
from hmtest.ml.utils import save_to_yaml
from typing_extensions import Annotated


def train(
    meta_path: Annotated[Path, typer.Argument(help="path to meta-data csv file")],
    image_root_path: Annotated[Path, typer.Argument(help="root path to image files")],
    out_root_path: Annotated[Path, typer.Argument(help="output path", exists=True)],
    exp_name: Annotated[
        str, typer.Argument(help="name of experiment (used for logging)")
    ],
    image_size: Annotated[int, typer.Option(help="size of input image")] = 512,
    batch_size: Annotated[int, typer.Option()] = 16,
    learning_rate: Annotated[float, typer.Option()] = 1e-4,
    weight_decay: Annotated[float, typer.Option()] = 0,
    checkpoint_period: Annotated[int, typer.Option()] = 1,
    n_epochs: Annotated[int, typer.Option()] = 4,
    seed: Annotated[int, typer.Option()] = 42,
    resume_cp_path: Annotated[
        Optional[Path], typer.Option(help="checkpoint to resume from")
    ] = None,
):
    """
    Training routine.
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    run_path = out_root_path / (current_time + "-" + exp_name)
    run_path.mkdir(exist_ok=True)

    save_to_yaml(run_path / "cfg.yaml", locals())

    import tensorflow as tf
    from hmtest.ml.model import BreastClassifier
    from hmtest.ml.trainer import Trainer
    from keras.losses import BinaryCrossentropy
    from keras.optimizers import Adam

    model = BreastClassifier(input_shape=(512, 512, 1))
    optim = Adam(learning_rate=learning_rate, weight_decay=weight_decay)
    criterion = BinaryCrossentropy(from_logits=True, reduction="sum")
    dataloaders = make_dataloaders(
        image_root_path,
        meta_path,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )
    trainer = Trainer(model, optim, criterion)

    tboard_train_writer = tf.summary.create_file_writer(str(run_path / "log" / "train"))
    tboard_val_writer = tf.summary.create_file_writer(str(run_path / "log" / "val"))
    train_clbks = make_callbacks(
        tboard_train_writer,
        model,
        mode="train",
        checkpoint_root_path=run_path / "checkpoints",
        checkpoint_period=checkpoint_period,
    )
    val_clbks = make_callbacks(tboard_val_writer, model, mode="val")

    for e in range(n_epochs):
        print(f"Epoch {e+1}/{n_epochs}")
        trainer.train_one_epoch(dataloaders["train"], callbacks=train_clbks)
        trainer.eval_one_epoch(dataloaders["val"], callbacks=val_clbks)
