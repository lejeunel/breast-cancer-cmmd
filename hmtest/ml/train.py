from pathlib import Path
from typing import Optional

import typer
import datetime
from typing_extensions import Annotated
from hmtest.ml.utils import save_to_yaml


def make_dataloaders(
    image_root_path: Path,
    meta_path: Path,
    batch_size: int,
    image_size: int,
    splits=["train", "val"],
    seed=42,
):
    from hmtest.ml.dataloader import DataLoader

    dataloaders = {
        s: DataLoader(
            image_root_path,
            meta_path,
            split=s,
            batch_size=batch_size,
            image_size=image_size,
        )
        for s in ["train", "val"]
    }

    return dataloaders


def train(
    meta_path: Annotated[Path, typer.Argument(help="path to meta-data csv file")],
    image_root_path: Annotated[Path, typer.Argument(help="root path to image files")],
    out_path: Annotated[Path, typer.Argument(help="output path", exists=True)],
    image_size: Annotated[int, typer.Option(help="size of input image")] = 512,
    batch_size: Annotated[int, typer.Option()] = 16,
    learning_rate: Annotated[float, typer.Option()] = 1e-3,
    weight_decay: Annotated[float, typer.Option()] = 0,
    n_epochs: Annotated[int, typer.Option()] = 200,
    seed: Annotated[int, typer.Option()] = 42,
    resume_cp_path: Annotated[
        Optional[Path], typer.Option(help="checkpoint to resume from")
    ] = None,
    pretrained: Annotated[Optional[bool], typer.Option()] = True,
):
    """
    Training routine.
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    run_path = out_path / current_time
    run_path.mkdir(exist_ok=True)

    save_to_yaml(run_path / "cfg.yaml", locals())

    from hmtest.ml.model import make_model
    from hmtest.ml.trainer import Trainer
    from hmtest.ml.callbacks import BatchLossWriterCallback
    from keras.optimizers import Adam
    from keras.losses import BinaryCrossentropy
    import tensorflow as tf

    model = make_model(input_shape=(512, 512, 1))
    optim = Adam(learning_rate=learning_rate, weight_decay=weight_decay)
    criterion = BinaryCrossentropy()
    dataloaders = make_dataloaders(
        image_root_path,
        meta_path,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )
    trainer = Trainer(model, optim, criterion)

    tboard_writer = tf.summary.create_file_writer(str(run_path))
    post_batch_clbks = [BatchLossWriterCallback(tboard_writer)]

    for e in range(n_epochs):
        print(f"Epoch {e+1}/{n_epochs}")
        trainer.train_one_epoch(
            dataloaders["train"], post_batch_callbacks=post_batch_clbks
        )
