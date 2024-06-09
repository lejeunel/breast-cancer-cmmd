import tensorflow as tf
from hmtest.ml.dataloader import Batch


class BatchCallback:
    def on_epoch_end(self):
        pass


class BatchLossWriterCallback(BatchCallback):
    def __init__(self, writer, out_field, batch_field):
        self.writer = writer
        self.out_field = out_field
        self.batch_field = batch_field

    def __call__(self, batch: Batch, step: int):

        with self.writer.as_default():
            tf.summary.scalar(
                self.out_field, getattr(batch, self.batch_field), step=step
            )


class BatchMetricWriterCallback(BatchCallback):
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

    def __call__(self, batch: Batch, step: int):

        with self.writer.as_default():
            self.metric_fn.update_state(
                getattr(batch, self.batch_field_true),
                getattr(batch, self.batch_field_pred),
            )
            tf.summary.scalar(
                self.out_field, self.metric_fn.result().numpy()[0], step=step
            )

    def on_epoch_end(self):
        self.metric_fn.reset_state()
