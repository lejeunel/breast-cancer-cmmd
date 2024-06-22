from pathlib import Path

from breastclf.ml.shared import Batch
from breastclf.ml.metrics import BinaryPrecisionAtFixedRecall
import torch
from torcheval import metrics

import pandas as pd


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
        metric_mode="maximize",
        metrics=[],
    ):
        """
        root_path: Path where checkpoints are saved
        model[nn.Module]: Model to save
        optimizer[torch.Optimizer]: Holds state of optimizer (gradients, ...)
        metric_fn[torcheval.Metric]: Function that computes performance metric
        batch_field_pred[str]: attribute name of Batch object with predictions
        batch_field_true[str]: attribute name of Batch object with true values
        epoch_period[int]: Save checkpoint period
        metric_mode[str]: Whether our best model must maximize or minimize the provided
                            metric
        """
        self.root_path = root_path
        self.epoch = 1
        self.epoch_period = epoch_period
        self.model = model
        self.optimizer = optimizer
        self.metric_fn = metric_fn
        self.batch_field_pred = batch_field_pred
        self.batch_field_true = batch_field_true

        assert metric_mode in [
            "maximize",
            "minimize",
        ], f"metric_mode must be in ['maximize', 'minimize']"
        self.metric_mode = metric_mode
        self.metrics = metrics

        self.root_path.mkdir(exist_ok=True)

    def on_batch_end(self, batch: Batch, *args, **kwargs):
        meta_ = batch.meta.drop_duplicates("breast_id")
        y_pred = getattr(batch, self.batch_field_pred).flatten()[meta_.index]
        y_true = getattr(batch, self.batch_field_true).flatten().int()[meta_.index]
        self.metric_fn.update(y_pred, y_true)

    def _save_checkpoint(self, path: Path, with_optimizer: bool = False):
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": (
                    self.optimizer.state_dict() if with_optimizer else None
                ),
                "metric": self.metric_fn.compute(),
                "metrics": self.metrics,
            },
            path,
        )
        print(f"saved checkpoint to {path}")

    def on_epoch_end(self, *args, **kwargs):

        if (self.epoch % self.epoch_period) == 0:
            self._save_checkpoint(self.root_path / "last.pth.tar", with_optimizer=True)

        new_metric = self.metric_fn.compute()
        do_save_as_best = False
        if len(self.metrics) == 0:
            do_save_as_best = True
        elif self.metric_mode == "maximize":
            if new_metric > max(self.metrics):
                do_save_as_best = True
        else:
            if new_metric < min(self.metrics):
                do_save_as_best = True

        if do_save_as_best:
            self._save_checkpoint(self.root_path / "best.pth.tar")

        self.metrics.append(new_metric)

        self.epoch += 1
        self.metric_fn.reset()


class LossWriterCallback(Callback):
    def __init__(self, writer, out_field, batch_field):
        self.writer = writer
        self.out_field = out_field
        self.batch_field = batch_field

    def on_batch_end(self, batch: Batch, *args, **kwargs):

        self.writer.add_scalar(
            self.out_field, getattr(batch, self.batch_field), batch.iter
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

        meta_ = batch.meta.drop_duplicates("breast_id")
        y_pred = getattr(batch, self.batch_field_pred).flatten()[meta_.index]
        y_true = getattr(batch, self.batch_field_true).flatten().int()[meta_.index]
        scalar = self.metric_fn.update(y_pred, y_true).compute()

        self.writer.add_scalar(self.out_field, scalar, batch.iter)

    def on_epoch_end(self, *args, **kwargs):
        # print(f"MetricWriterCallbacks / {self.metric_fn}: {self.metric_fn.compute()}")
        self.metric_fn.reset()


class PerImageCallback(Callback):
    """ """

    def __init__(self, out_path: Path, batch_field_true, batch_field_pred):
        self.batch_field_pred = batch_field_pred
        self.batch_field_true = batch_field_true

        self.out_path = out_path

        self.all_meta = []

    def on_batch_end(self, batch: Batch, *args, **kwargs):

        meta_ = batch.meta.copy()
        y_pred = getattr(batch, self.batch_field_pred).flatten().detach().cpu().numpy()
        y_true = (
            getattr(batch, self.batch_field_true).flatten().int().detach().cpu().numpy()
        )

        meta_["_pred"] = y_pred
        meta_["_true"] = y_true

        self.all_meta.append(meta_)

    def on_epoch_end(self, epoch):
        all_meta = pd.concat(self.all_meta)
        all_meta.to_csv(self.out_path / f"ep_{epoch:03d}.csv")


def make_callbacks(
    tboard_writer,
    binarizer_type,
    model=None,
    optimizer=None,
    checkpoint_root_path=None,
    checkpoint_period=1,
    mode="train",
    past_metrics=[],
):
    callbacks = [
        MetricWriterCallback(
            tboard_writer,
            metrics.BinaryAccuracy(threshold=0.5),
            "acc",
            "tgt_type",
            "pred_type",
        ),
        MetricWriterCallback(
            tboard_writer,
            metrics.BinaryAUROC(),
            "auc_roc",
            "tgt_type",
            "pred_type",
        ),
        MetricWriterCallback(
            tboard_writer,
            metrics.BinaryF1Score(threshold=0.5),
            "f1_type",
            "tgt_type",
            "pred_type",
        ),
        MetricWriterCallback(
            tboard_writer,
            BinaryPrecisionAtFixedRecall(min_recall=1.0),
            "precision_at_recall_1",
            "tgt_type",
            "pred_type",
        ),
    ]

    if mode == "train":
        callbacks += [
            LossWriterCallback(tboard_writer, "loss_abnorm", "loss_abnorm"),
            LossWriterCallback(tboard_writer, "loss_type", "loss_type"),
        ]

    if mode == "val":
        callbacks += [
            ModelCheckpointCallback(
                root_path=checkpoint_root_path,
                model=model,
                optimizer=optimizer,
                epoch_period=checkpoint_period,
                metric_fn=metrics.BinaryAUROC(),
                batch_field_pred="pred_type",
                batch_field_true="tgt_type",
                metrics=past_metrics,
            ),
            PerImageCallback(
                out_path=checkpoint_root_path,
                batch_field_true="tgt_type",
                batch_field_pred="pred_type",
            ),
        ]

    return callbacks
