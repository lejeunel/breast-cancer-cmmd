from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch
from hmtest.ml import image_preprocessing as impp


@dataclass
class Batch:
    """Container of input/outputs that
    we pass into trainer and callbacks"""

    images: list[np.array]
    meta: pd.DataFrame

    iter: int = 0

    pred_type: Optional[list[float]] = None
    pred_type_post: Optional[list[float]] = None
    pred_abnorm: Optional[list[list[float]]] = None

    tgt_type: Optional[list[int]] = None
    tgt_abnorm: Optional[list[list[int]]] = None

    loss_abnorm: Optional[float] = None
    loss_type: Optional[float] = None
    loss_type_post: Optional[float] = None

    def set_losses(self, losses: dict[torch.Tensor]):
        self.loss_type = losses["type"].detach()
        self.loss_type_post = losses["type_post"].detach()
        self.loss_abnorm = losses["abnorm"].detach()

    def set_predictions(self, logits: dict[torch.Tensor]):
        self.pred_type = logits["type"].sigmoid().detach()
        self.pred_type_post = logits["type_post"].sigmoid().detach()
        self.pred_abnorm = logits["abnorm"].sigmoid().detach()

    def to(self, device):
        self.tgt_type = self.tgt_type.to(device)
        self.tgt_abnorm = self.tgt_abnorm.to(device)
        self.images = self.images.to(device)


def _encode_binary_target(
    meta,
    in_field="classification",
    out_field="t_type",
    map_={"Benign": 0, "Malignant": 1},
):
    """
    Encode a categorical variable to binary using mapping.
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


class BreastDataset(Dataset):
    """ """

    def __init__(
        self,
        root_image_path: Path,
        meta_path: Path,
        split: str,
        image_size: int = 512,
        return_orig_image=False,
    ):
        """ """

        super().__init__()
        self.meta = pd.read_csv(meta_path)
        self.meta = self.meta.loc[self.meta.split == split].reset_index()

        self.meta = _encode_binary_target(self.meta)
        self.meta = _encode_multi_target(self.meta)

        self.root_image_path = root_image_path
        self.n_samples = len(self.meta)
        self.image_size = image_size

        self.return_orig_image = return_orig_image

    def __len__(self):
        return len(self.meta)

    def _read_and_resize_image(self, image_path):
        """
        Reads an image from disk and resize it
        """

        image = imread(image_path)

        if any([dim != self.image_size for dim in image.shape[:2]]):
            image = resize(
                image,
                (self.image_size, self.image_size),
                preserve_range=True,
                anti_aliasing=True,
            )

        return image.astype(np.uint8)

    def _preprocess_image(self, image: np.ndarray, is_left: bool):
        """
        1. Clean background noise
        2. Vertical mirroring to make all breast located on left side of frame
        3. Crop height to breast
        4. Resize to original size
        """

        orig_size = image.shape

        # Clean background noise
        mask = impp.triangle_thresholding(image)
        image[mask == 0] = 0

        # mirroring
        if not is_left:
            image = image[:, ::-1]

        # Crop height to breast
        n_pixels = mask.sum(axis=0)
        max_col_idx = np.argmax(n_pixels)
        max_col = mask[:, max_col_idx]
        nz_rows = np.where(max_col)[0]
        first_nz_row, last_nz_row = nz_rows[0], nz_rows[-1]
        image = image[first_nz_row : last_nz_row + 1, :]

        # resize to original size
        image = resize(
            image, orig_size, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)

        return image

    def get_ratio(self, field="t_type"):
        n_pos = (self.meta[field] == 1).sum()
        return n_pos / len(self.meta)

    def __getitem__(self, i):

        meta = self.meta.iloc[i]
        image_path = (
            Path(self.root_image_path)
            / meta.serie_id
            / (meta.filename.split(".")[0] + ".png")
        )

        orig_image = self._read_and_resize_image(image_path)[..., None]
        image = self._preprocess_image(orig_image.copy(), meta.side == "left")

        res = {"image": image, "meta": meta}

        if self.return_orig_image:
            res.update({"orig_image": orig_image})

        return res

    @staticmethod
    def collate_fn(samples: list[dict]):
        """
        convert list of samples obtained through __getitem__ to
        pytorch tensors
        """

        meta = pd.DataFrame([s["meta"] for s in samples])
        images = torch.tensor(
            np.stack([np.moveaxis(s["image"], 2, 0) for s in samples])
        ).float()

        t_type = torch.tensor(meta["t_type"].values)[..., None]

        t_abnorm = torch.tensor(meta["t_abnormality_calcification"].values)[..., None]

        return Batch(
            **{
                "meta": pd.DataFrame([s["meta"] for s in samples]),
                "images": images,
                "tgt_type": t_type.float(),
                "tgt_abnorm": t_abnorm.float(),
            }
        )


def make_dataloaders(
    image_root_path: Path,
    meta_path: Path,
    batch_size: int,
    image_size: int,
    n_workers: int = 8,
    n_batches_per_epoch=30,
):

    splits = ["train", "val", "test"]

    datasets = {
        s: BreastDataset(
            image_root_path,
            meta_path,
            split=s,
            image_size=image_size,
        )
        for s in splits
    }

    dataloaders = {
        s: DataLoader(
            d,
            batch_size=batch_size,
            num_workers=n_workers,
            collate_fn=BreastDataset.collate_fn,
            sampler=(
                RandomSampler(d, num_samples=n_batches_per_epoch * batch_size)
                if s == "train"
                else None
            ),
        )
        for s, d in datasets.items()
    }

    return dataloaders


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dset = BreastDataset(
        "data/png",
        meta_path="data/meta-images-split.csv",
        split="train",
        image_size=512,
        return_orig_image=True,
    )

    to_plot = [16, 0]
    fig, axes = plt.subplots(nrows=len(to_plot), ncols=2)

    for row, s_idx in enumerate(to_plot):
        sample = dset[s_idx]
        print(sample["meta"])
        axes[row][0].title.set_text("Original")
        axes[row][0].imshow(sample["orig_image"])
        axes[row][1].title.set_text("Preprocessed")
        axes[row][1].imshow(sample["image"])
    plt.show()
