import tensorflow.keras.ops
import tensorflow as tf
from tqdm import tqdm
import tensorflow as tf
import numpy as np


class Trainer:
    """
    Wrapper class used in training and validation routine
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        loss_factors={"type_pre": 1, "type_post": 1, "abnorm": 1},
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_factors = loss_factors

        self.train_epoch = 1
        self.train_iter = 0
        self.val_epoch = 1
        self.val_iter = 0

    def _forward(self, batch, mode):
        logits = self.model(batch.images, training=True if mode == "train" else False)

        batch.pred_pre_type = tf.keras.ops.sigmoid(logits["type_pre"])
        batch.pred_post_type = tf.keras.ops.sigmoid(logits["type_post"])
        batch.pred_abnorm = tf.keras.ops.sigmoid(logits["abnorm"])

        return batch, logits

    def _compute_losses(self, batch, logits, sample_weights):
        losses = {}
        losses["type_pre"] = self.criterion(
            batch.tgt_type, logits["type_pre"], sample_weights
        )
        losses["type_post"] = self.criterion(
            batch.tgt_type, logits["type_post"], sample_weights
        )
        losses["abnorm"] = self.criterion(batch.tgt_abnorm, logits["abnorm"])

        batch.loss_pre_type = losses["type_pre"].numpy()
        batch.loss_post_type = losses["type_post"].numpy()
        batch.loss_abnorm = losses["abnorm"].numpy()

        return batch, losses

    def train_one_epoch(self, dataloader, callbacks=[]):

        class_weights = dataloader.get_class_weights()

        for batch in (pbar := tqdm(dataloader)):
            with tf.GradientTape() as tape:

                sample_weights = class_weights[batch.meta.t_type]

                batch, logits = self._forward(batch, mode="train")
                batch, losses = self._compute_losses(batch, logits, sample_weights)

                total_loss = tensorflow.keras.ops.sum(
                    [v * losses[k] for k, v in self.loss_factors.items()]
                )

            gradients = tape.gradient(total_loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

            pbar.set_description(f"[train] lss: {total_loss.numpy().sum():.2e}")

            batch.iter = self.train_iter

            callbacks.on_batch_end(batch)

            self.train_iter += 1

        callbacks.on_epoch_end(epoch=self.train_epoch)

        self.train_epoch += 1

    def eval_one_epoch(self, dataloader, callbacks=[]):

        for batch in (pbar := tqdm(dataloader)):

            batch, losses = self._forward(batch, mode="val")
            total_loss = tensorflow.keras.ops.sum(
                [v * losses[k] for k, v in self.loss_factors.items()]
            )

            pbar.set_description(f"[val] lss: {total_loss.numpy().sum():.2e}")

            batch.iter = self.val_iter

            callbacks.on_batch_end(batch)

            self.val_iter += 1

        callbacks.on_epoch_end(epoch=self.val_epoch)

        self.val_epoch += 1
