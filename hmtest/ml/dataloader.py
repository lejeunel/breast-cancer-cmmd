from pathlib import Path
import tensorflow as tf


class DataLoader(tf.keras.utils.Sequence):
    """ """

    def __init__(
        self,
        root_image_path: Path,
        meta_path: Path,
        split: str,
        batch_size: int = 1,
        image_size: int = 512,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """ """
        import pandas as pd
        import numpy as np

        super().__init__()
        self.meta = pd.read_csv(meta_path)
        self.meta = self.meta.loc[self.meta.split == split].reset_index()

        self.meta = self.meta.loc[~self.meta.classification.isna()]
        self.meta["target"] = [
            1 if r.classification == "Malignant" else 0 for _, r in self.meta.iterrows()
        ]

        self.root_image_path = root_image_path
        self.n_samples = len(self.meta)
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.sample_indices = np.arange(self.n_samples)
        self.n_batches = self.n_samples // self.batch_size

        np.random.seed(seed)
        self.epoch_batch_indices = self._epoch_batch_indices()

    def __len__(self):
        """
        number of batches the generator can produce
        """
        return len(self.meta) // self.batch_size

    def _epoch_batch_indices(self):
        """
        returns a n_batches x batch_size array with the indices for each
        batch for an epoch
        """
        import numpy as np

        if self.shuffle:
            np.random.shuffle(self.sample_indices)

        epoch_indices = self.sample_indices[: self.n_batches * self.batch_size]
        epoch_batch_indices = np.reshape(
            epoch_indices, (self.n_batches, self.batch_size)
        )

        return epoch_batch_indices

    def _prepare_image(self, image_path):
        """
        Reads an image from disk and transform it prior
        to feeding to batch
        """
        from skimage.io import imread
        from skimage.transform import resize

        image = imread(image_path)

        if any([dim != self.image_size for dim in image.shape[:2]]):
            image = resize(
                image,
                (self.image_size, self.image_size),
                preserve_range=True,
                anti_aliasing=True,
            )

        # image = image / 255
        # image = (image * 2) - 1
        image = image[..., None]

        return image

    def __getitem__(self, batch_ind):
        """
        generates a batch of data
        """

        import numpy as np
        import pandas as pd

        images = []
        targets = []
        metas = []
        for ind in self.epoch_batch_indices[batch_ind]:
            meta = self.meta.iloc[ind]
            image_path = (
                self.root_image_path
                / meta.serie_id
                / (meta.filename.split(".")[0] + ".png")
            )

            image = self._prepare_image(image_path)

            images.append(image)
            targets.append(meta.target)
            metas.append(meta)
        return {
            "image": np.array(images),
            "target": np.array(targets),
            "meta": pd.DataFrame(metas),
        }

    def on_epoch_end(self):
        """
        maybe reshuffle after epoch
        """
        import numpy as np

        self.sample_indices = np.arange(self.n_samples)
        self.epoch_batch_indices = self._epoch_batch_indices()


if __name__ == "__main__":
    from pathlib import Path

    root_path = Path("data")
    dl_train = DataLoader(
        root_path / "png",
        root_path / "meta-images-split.csv",
        "train",
        batch_size=4,
        image_size=320,
    )
    batch = dl_train[0]
