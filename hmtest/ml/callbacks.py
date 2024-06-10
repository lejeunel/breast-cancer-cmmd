from pathlib import Path

import tensorflow as tf
from hmtest.ml.dataloader import Batch
from keras import metrics
from keras.callbacks import Callback, CallbackList


class ModelCheckpointCallback(Callback):
    def __init__(self, root_path: Path, model, epoch_period=2):
        self.root_path = root_path
        self.epoch = 0
        self.epoch_period = epoch_period
        self._model = model

        self.root_path.mkdir(exist_ok=True)

    def on_epoch_end(self, *args, **kwargs):

        if self.epoch % self.epoch_period:
            cp_path = self.root_path / f"ep_{self.epoch:03d}.weights.h5"
            print(f"saving checkpoint to {cp_path}...")
            self._model.save_weights(cp_path)

        self.epoch += 1


class LossWriterCallback(Callback):
    def __init__(self, writer, out_field, batch_field):
        self.writer = writer
        self.out_field = out_field
        self.batch_field = batch_field

    def on_batch_end(self, batch: Batch, *args, **kwargs):

        with self.writer.as_default():
            tf.summary.scalar(
                self.out_field, getattr(batch, self.batch_field), step=batch.iter
            )


class MetricWriterCallback(Callback):
    """
    Compute a running metric with a callable derived from
    keras.metrics.Metric
    """

    def __init__(
        self, writer, metric_fn, out_field, batch_field_true, batch_field_pred
    ):
        self.metric_fn = metric_fn
        self.writer = writer
        self.out_field = out_field
        self.batch_field_true = batch_field_true
        self.batch_field_pred = batch_field_pred

    def on_batch_end(self, batch: Batch, *args, **kwargs):

        with self.writer.as_default():
            self.metric_fn.update_state(
                getattr(batch, self.batch_field_true),
                getattr(batch, self.batch_field_pred),
            )
            tf.summary.scalar(
                self.out_field, self.metric_fn.result().numpy()[0], step=batch.iter
            )

    def on_epoch_end(self, *args, **kwargs):
        self.metric_fn.reset_state()


def make_callbacks(tboard_writer, model, checkpoint_root_path=None, mode="train"):
    callbacks = [
        LossWriterCallback(tboard_writer, "loss_abnorm", "loss_abnorm"),
        LossWriterCallback(tboard_writer, "loss_pre_diagn", "loss_pre_diagn"),
        LossWriterCallback(tboard_writer, "loss_post_diagn", "loss_post_diagn"),
        MetricWriterCallback(
            tboard_writer,
            metrics.F1Score(threshold=0.5),
            "f1_pre_diagn",
            "tgt_diagn",
            "pred_pre_diagn",
        ),
        MetricWriterCallback(
            tboard_writer,
            metrics.F1Score(threshold=0.5),
            "f1_post_diagn",
            "tgt_diagn",
            "pred_post_diagn",
        ),
        MetricWriterCallback(
            tboard_writer,
            metrics.F1Score(threshold=0.5),
            "f1_abnorm",
            "tgt_abnorm",
            "pred_abnorm",
        ),
    ]

    if mode == "train":
        callbacks += [
            ModelCheckpointCallback(root_path=checkpoint_root_path, model=model)
        ]

    callbacks = CallbackList(callbacks=callbacks)

    return callbacks
