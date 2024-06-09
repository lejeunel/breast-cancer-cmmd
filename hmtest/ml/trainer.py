from tqdm import tqdm
from hmtest.ml.callbacks import Batch


class Trainer:
    """
    Wrapper class used in training and validation routine
    """

    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.epoch = 0
        self.iter = 0

    def train_one_epoch(
        self, dataloader, post_batch_callbacks=[], post_epoch_callbacks=[]
    ):

        import tensorflow as tf

        class_weights = dataloader.get_class_weights()

        breakpoint()
        for s in (pbar := tqdm(dataloader)):
            with tf.GradientTape() as tape:

                images = s["image"]
                logits = self.model(images, training=True)

                sample_weights = class_weights[s["target"]]

                loss = self.criterion(s["target"], logits, sample_weight=sample_weights)

                result = Batch(
                    s["image"],
                    tf.math.sigmoid(logits).numpy(),
                    s["target"][..., None],
                    loss,
                )

            gradients = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

            pbar.set_description(f"[train] lss: {loss.numpy().sum():.2e}")

            for clbk in post_batch_callbacks:
                clbk(result, self.iter)

            self.iter += 1

        for clkb in post_batch_callbacks:
            clbk.on_epoch_end()

        self.epoch += 1
