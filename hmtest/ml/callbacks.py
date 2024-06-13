from pathlib import Path

from hmtest.ml.dataloader import Batch
from hmtest.ml.metrics import BinaryPrecisionAtFixedRecall
import torch
from torcheval import metrics
from sklearn.metrics import ConfusionMatrixDisplay, multilabel_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


class Callback:
    def on_epoch_end(self, *args, **kwargs):
        pass

    def on_batch_end(self, *args, **kwargs):
        pass


class ModelCheckpointCallback(Callback):
    def __init__(
        self,
        root_path: Path,
        model,
        optimizer,
        metric_fn,
        batch_field_pred,
        batch_field_true,
        epoch_period=1,
    ):
        self.root_path = root_path
        self.epoch = 1
        self.epoch_period = epoch_period
        self.model = model
        self.optimizer = optimizer
        self.metric_fn = metric_fn
        self.batch_field_pred = batch_field_pred
        self.batch_field_true = batch_field_true

        self.root_path.mkdir(exist_ok=True)

    def on_batch_end(self, batch: Batch, *args, **kwargs):

        self.metric_fn.update(
            getattr(batch, self.batch_field_pred).flatten(),
            getattr(batch, self.batch_field_true).flatten().int(),
        )

    def on_epoch_end(self, *args, **kwargs):

        if (self.epoch % self.epoch_period) == 0:
            path_ = self.root_path / f"epoch_{self.epoch:03d}.pth.tar"
            torch.save(
                {
                    "epoch": self.epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metric": self.metric_fn.compute(),
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
        out_field,
        batch_field_true,
        batch_field_pred,
        binarizer,
        threshold=0.5,
    ):
        self.writer = writer
        self.out_field = out_field
        self.batch_field_true = batch_field_true
        self.batch_field_pred = batch_field_pred
        self.binarizer = binarizer
        self.threshold = 0.5

        self.y = []
        self.y_true = []

    def on_batch_end(self, batch: Batch, *args, **kwargs):

        self.y += [
            getattr(batch, self.batch_field_pred).squeeze().detach().cpu().numpy()
        ]
        self.y_true += [
            getattr(batch, self.batch_field_true).squeeze().detach().cpu().int().numpy()
        ]

    def on_epoch_end(self, epoch, *args, **kwargs):

        self.y = np.concatenate(self.y) >= self.threshold
        self.y_true = np.concatenate(self.y_true)

        assert (
            self.y.ndim == self.y_true.ndim
        ), f"target and predictor dimensions do not match: {self.y.ndim}, and {self.y_true.ndim}!"

        is_binary = True
        n_predictors = 1
        if self.y.ndim > 1 and self.y_true.ndim > 1:
            is_binary = False
            n_predictors = self.y.shape[-1]

        fig, axes = plt.subplots(nrows=1, ncols=n_predictors)

        if is_binary:
            y = self.binarizer.inverse_transform(self.y)
            y_true = self.binarizer.inverse_transform(self.y_true)
            ConfusionMatrixDisplay.from_predictions(
                y_true, y, labels=self.binarizer.classes_, ax=axes, normalize="true"
            )
        else:
            mat = multilabel_confusion_matrix(self.y_true, self.y)
            for i, (ax, m) in enumerate(zip(axes, mat)):
                class_ = self.binarizer.classes_[i]
                cmd = ConfusionMatrixDisplay(
                    confusion_matrix=m,
                    display_labels=["False", "True"],
                )
                ax.title.set_text(class_)
                cmd.plot(ax=ax)

        self.writer.add_figure(self.out_field, fig, epoch)
        plt.tight_layout()
        plt.close()

        # reset state
        self.y = []
        self.y_true = []


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
    binarizer_type,
    binarizer_abnorm,
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
            metrics.BinaryAUROC(),
            "auc_roc_pre",
            "tgt_type",
            "pred_type",
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
        ConfusionMatrixWriterCallback(
            tboard_writer, "conf_mat", "tgt_type", "pred_type", binarizer=binarizer_type
        ),
        ConfusionMatrixWriterCallback(
            tboard_writer,
            "conf_mat_abnorm",
            "tgt_abnorm",
            "pred_abnorm",
            binarizer=binarizer_abnorm,
        ),
    ]

    if mode == "val":
        callbacks += [
            ModelCheckpointCallback(
                root_path=checkpoint_root_path,
                model=model,
                optimizer=optimizer,
                epoch_period=checkpoint_period,
                metric_fn=metrics.BinaryAUROC(),
                batch_field_pred="tgt_type",
                batch_field_true="pred_type",
            )
        ]

    return callbacks
