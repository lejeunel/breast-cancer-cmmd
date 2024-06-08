from dataclasses import dataclass
import numpy as np


@dataclass
class Batch:
    """Simple container of results that we pass into callbacks"""

    images: list[np.array]
    predictions: list[float]
    loss: float


class BatchLossWriterCallback:
    def __init__(self, writer):
        self.writer = writer

    def __call__(self, batch: Batch, step: int):
        import tensorflow as tf

        with self.writer.as_default():
            tf.summary.scalar("loss", batch.loss, step=step)
