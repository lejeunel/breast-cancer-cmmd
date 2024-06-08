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

        for s in (pbar := tqdm(dataloader)):
            with tf.GradientTape() as tape:

                images = s["image"]
                pred_probas = self.model(images, training=True)

                loss = self.criterion(s["target"], pred_probas)

                result = Batch(s["image"], pred_probas, loss)
                print(s["meta"])
                print("min: {}".format([im.min() for im in images]))
                print("max: {}".format([im.max() for im in images]))
                print(pred_probas)

            gradients = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

            pbar.set_description(f"[train] lss: {loss.numpy().sum():.2E}")

            for clbk in post_batch_callbacks:
                clbk(result, self.iter)

            self.iter += 1

        self.epoch += 1
