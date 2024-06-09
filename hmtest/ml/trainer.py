from tqdm import tqdm
import keras.ops
from hmtest.ml.callbacks import Batch


class Trainer:
    """
    Wrapper class used in training and validation routine
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        loss_factors={"diagn_pre": 1, "diagn_post": 1, "abnorm": 1},
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_factors = loss_factors

        self.epoch = 0
        self.iter = 0

    def train_one_epoch(
        self, dataloader, post_batch_callbacks=[], post_epoch_callbacks=[]
    ):

        import tensorflow as tf

        for batch in (pbar := tqdm(dataloader)):
            with tf.GradientTape() as tape:

                logits = self.model(batch.images, training=True)

                batch.pred_pre_diagn = tf.sigmoid(logits["diagn_pre"])
                batch.pred_post_diagn = tf.sigmoid(logits["diagn_post"])
                batch.pred_abnorm = tf.sigmoid(logits["abnorm"])

                losses = {}
                losses["diagn_pre"] = self.criterion(
                    batch.tgt_diagn, logits["diagn_pre"]
                )
                losses["diagn_post"] = self.criterion(
                    batch.tgt_diagn, logits["diagn_post"]
                )
                losses["abnorm"] = self.criterion(batch.tgt_abnorm, logits["abnorm"])

                total_loss = keras.ops.sum(
                    [v * losses[k] for k, v in self.loss_factors.items()]
                )

            gradients = tape.gradient(total_loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

            pbar.set_description(f"[train] lss: {total_loss.numpy().sum():.2e}")

            batch.loss_pre_diagn = losses["diagn_pre"].numpy()
            batch.loss_post_diagn = losses["diagn_post"].numpy()
            batch.loss_abnorm = losses["abnorm"].numpy()

            for clbk in post_batch_callbacks:
                clbk(batch, self.iter)

            self.iter += 1

        for clkb in post_batch_callbacks:
            clbk.on_epoch_end()

        self.epoch += 1
