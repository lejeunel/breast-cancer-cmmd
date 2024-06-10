from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize


@dataclass
class Batch:
    """Simple container of input/outputs that
    we pass into model and callbacks"""

    images: list[np.array]
    meta: pd.DataFrame

    iter: int = 0

    pred_pre_diagn: Optional[list[float]] = None
    pred_post_diagn: Optional[list[float]] = None
    pred_abnorm: Optional[list[float]] = None

    tgt_diagn: Optional[list[int]] = None
    tgt_abnorm: Optional[list[list[int]]] = None

    loss_abnorm: Optional[float] = None
    loss_pre_diagn: Optional[float] = None
    loss_post_diagn: Optional[float] = None


def _encode_binary_target(
    meta,
    in_field="classification",
    out_field="t_diagn",
    map_={"Benign": 0, "Malignant": 1},
):
    """
    Encode diagnosis target variable
    """
    values = [map_[r[in_field]] for _, r in meta.iterrows()]
    meta[out_field] = values

    return meta


def _encode_multi_target(
    meta,
    in_field="abnormality",
    out_field_prefix="t_abnormality",
    universal_value="both",
):
    """
    Encode abnormality target with indicator variables using prefix "out_field".
    Use universal_value to specify which input value corresponds to all other values
    """
    values = [v for v in meta[in_field].unique() if v != universal_value]
    new_cols = [out_field_prefix + "_" + v for v in values]

    for v, c in zip(values, new_cols):
        meta[c] = 0
        meta[c] = [1 if r[in_field] == v else 0 for _, r in meta.iterrows()]

    meta.loc[meta[in_field] == universal_value, new_cols] = 1

    return meta


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

        super().__init__()
        self.meta = pd.read_csv(meta_path)
        self.meta = self.meta.loc[self.meta.split == split].reset_index()

        self.meta = _encode_binary_target(self.meta)
        self.meta = _encode_multi_target(self.meta)

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

        image = imread(image_path)

        if any([dim != self.image_size for dim in image.shape[:2]]):
            image = resize(
                image,
                (self.image_size, self.image_size),
                preserve_range=True,
                anti_aliasing=True,
            )

        image = image[..., None]

        return image

    def __getitem__(self, batch_ind):
        """
        generates a batch of data
        """

        images = []
        metas = []
        tgt_diagn = []
        tgt_abnorm = []
        for ind in self.epoch_batch_indices[batch_ind]:
            meta = self.meta.iloc[ind]
            image_path = (
                self.root_image_path
                / meta.serie_id
                / (meta.filename.split(".")[0] + ".png")
            )

            image = self._prepare_image(image_path)

            images.append(image)
            tgt_diagn.append(meta["t_diagn"])
            tgt_abnorm.append(meta[[k for k in meta.keys() if "t_abnorm" in k]])
            metas.append(meta)

        batch = Batch(
            images=np.array(images),
            meta=pd.DataFrame(metas),
            tgt_diagn=np.array(tgt_diagn)[..., None],
            tgt_abnorm=np.array(tgt_abnorm),
        )
        return batch

    def on_epoch_end(self):
        """
        maybe reshuffle after epoch
        """

        self.sample_indices = np.arange(self.n_samples)
        self.epoch_batch_indices = self._epoch_batch_indices()


def make_dataloaders(
    image_root_path: Path,
    meta_path: Path,
    batch_size: int,
    image_size: int,
    splits=["train", "val"],
    seed=42,
):

    dataloaders = {
        s: DataLoader(
            image_root_path,
            meta_path,
            split=s,
            batch_size=batch_size,
            image_size=image_size,
        )
        for s in ["train", "val"]
    }

    return dataloaders
