from pathlib import Path

from hmtest.ml.dataloader import Batch
from hmtest.ml.metrics import BinaryPrecisionAtFixedRecall
import torch
from torcheval import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Callback:
    def on_epoch_end(self, *args, **kwargs):
        pass

    def on_batch_end(self, *args, **kwargs):
        pass


class ModelCheckpointCallback(Callback):
    def __init__(self, root_path: Path, model, optimizer, epoch_period=2):
        self.root_path = root_path
        self.epoch = 1
        self.epoch_period = epoch_period
        self.model = model
        self.optimizer = optimizer

        self.root_path.mkdir(exist_ok=True)

    def on_epoch_end(self, *args, **kwargs):

        if (self.epoch % self.epoch_period) == 0:
            path_ = self.root_path / f"epoch_{self.epoch:03d}.pth.tar"
            torch.save(
                {
                    "epoch": self.epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                path_,
            )
            print(f"saved checkpoint to {path_}")

        self.epoch += 1


class LossWriterCallback(Callback):
    def __init__(self, writer, out_field, batch_field):
        self.writer = writer
        self.out_field = out_field
        self.batch_field = batch_field

    def on_batch_end(self, batch: Batch, *args, **kwargs):

        self.writer.add_scalar(
            self.out_field, getattr(batch, self.batch_field), batch.iter
        )


class ConfusionMatrixWriterCallback(Callback):

    def __init__(
        self,
        writer,
        metric_fn,
        out_field,
        batch_field_true,
        batch_field_pred,
        class_names=None,
    ):
        self.metric_fn = metric_fn
        self.writer = writer
        self.out_field = out_field
        self.batch_field_true = batch_field_true
        self.batch_field_pred = batch_field_pred
        self.class_names = class_names

    def on_batch_end(self, batch: Batch, *args, **kwargs):

        self.metric_fn.update(
            getattr(batch, self.batch_field_pred).flatten(),
            getattr(batch, self.batch_field_true).flatten().int(),
        )

    def on_epoch_end(self, epoch, *args, **kwargs):
        confusion_mat = self.metric_fn.compute().numpy()
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_mat, display_labels=self.class_names
        )
        disp.plot()
        self.writer.add_figure(self.out_field, plt.gcf(), epoch)
        self.metric_fn.reset()


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

        scalar = self.metric_fn.update(
            getattr(batch, self.batch_field_pred).flatten(),
            getattr(batch, self.batch_field_true).flatten().int(),
        ).compute()

        self.writer.add_scalar(self.out_field, scalar, batch.iter)

    def on_epoch_end(self, *args, **kwargs):
        self.metric_fn.reset()


def make_callbacks(
    tboard_writer,
    model=None,
    optimizer=None,
    checkpoint_root_path=None,
    checkpoint_period=1,
    mode="train",
):
    callbacks = [
        LossWriterCallback(tboard_writer, "loss_abnorm", "loss_abnorm"),
        LossWriterCallback(tboard_writer, "loss_pre_type", "loss_type"),
        LossWriterCallback(tboard_writer, "loss_post_type", "loss_type_post"),
        MetricWriterCallback(
            tboard_writer,
            metrics.BinaryAccuracy(threshold=0.5),
            "acc_pre",
            "tgt_type",
            "pred_type",
        ),
        MetricWriterCallback(
            tboard_writer,
            metrics.BinaryAccuracy(threshold=0.5),
            "acc_post",
            "tgt_type",
            "pred_type_post",
        ),
        MetricWriterCallback(
            tboard_writer,
            metrics.BinaryAUROC(),
            "auc_roc_pre",
            "tgt_type",
            "pred_type",
        ),
        MetricWriterCallback(
            tboard_writer,
            metrics.BinaryAUROC(),
            "auc_roc_post",
            "tgt_type",
            "pred_type_post",
        ),
        MetricWriterCallback(
            tboard_writer,
            metrics.BinaryF1Score(threshold=0.5),
            "f1_pre_type",
            "tgt_type",
            "pred_type",
        ),
        MetricWriterCallback(
            tboard_writer,
            metrics.BinaryF1Score(threshold=0.5),
            "f1_post_type",
            "tgt_type",
            "pred_type_post",
        ),
        MetricWriterCallback(
            tboard_writer,
            metrics.BinaryF1Score(threshold=0.5),
            "f1_abnorm",
            "tgt_abnorm",
            "pred_abnorm",
        ),
        MetricWriterCallback(
            tboard_writer,
            BinaryPrecisionAtFixedRecall(min_recall=1.0),
            "precision_at_recall_1_pre",
            "tgt_type",
            "pred_type",
        ),
        MetricWriterCallback(
            tboard_writer,
            BinaryPrecisionAtFixedRecall(min_recall=1.0),
            "precision_at_recall_1_post",
            "tgt_type",
            "pred_type_post",
        ),
        ConfusionMatrixWriterCallback(
            tboard_writer,
            metrics.BinaryConfusionMatrix(threshold=0.5, normalize="true"),
            "conf_mat",
            "tgt_type",
            "pred_type",
        ),
        ConfusionMatrixWriterCallback(
            tboard_writer,
            metrics.BinaryConfusionMatrix(threshold=0.5, normalize="true"),
            "conf_mat_post",
            "tgt_type",
            "pred_type_post",
        ),
        ConfusionMatrixWriterCallback(
            tboard_writer,
            metrics.BinaryConfusionMatrix(threshold=0.5, normalize="true"),
            "conf_mat_abnorm",
            "tgt_abnorm",
            "pred_abnorm",
        ),
    ]

    if mode == "train":
        callbacks += [
            ModelCheckpointCallback(
                root_path=checkpoint_root_path,
                model=model,
                optimizer=optimizer,
                epoch_period=checkpoint_period,
            )
        ]

    return callbacks
