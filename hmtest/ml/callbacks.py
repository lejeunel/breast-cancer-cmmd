import tensorflow as tf
from dataclasses import dataclass
import numpy as np


@dataclass
class Batch:
    """Simple container of results that we pass into callbacks"""

    images: list[np.array]
    predictions: list[float]
    targets: list[float]
    loss: float


class BatchCallback:
    def on_epoch_end(self):
        pass


class BatchLossWriterCallback(BatchCallback):
    def __init__(self, writer):
        self.writer = writer

    def __call__(self, batch: Batch, step: int):

        with self.writer.as_default():
            tf.summary.scalar("loss", batch.loss, step=step)


class BatchMetricWriterCallback(BatchCallback):
    """
    Compute a running metric with a callable derived from
    keras.metrics.Metric
    """

    def __init__(self, writer, metric_fn, field):
        self.metric_fn = metric_fn
        self.writer = writer
        self.field = field

    def __call__(self, batch: Batch, step: int):

        with self.writer.as_default():
            self.metric_fn.update_state(batch.targets, batch.predictions)
            tf.summary.scalar(self.field, self.metric_fn.result().numpy(), step=step)

    def on_epoch_end(self):
        self.metric_fn.reset_state()
