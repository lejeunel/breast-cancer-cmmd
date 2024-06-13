from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
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

    def set_predictions(self, predictions: dict[torch.Tensor]):
        self.pred_type = predictions["type"].detach()
        self.pred_type_post = predictions["type_post"].detach()
        self.pred_abnorm = predictions["abnorm"].detach()

    def to(self, device):
        self.tgt_type = self.tgt_type.to(device)
        self.tgt_abnorm = self.tgt_abnorm.to(device)
        self.images = self.images.to(device)


MAP_ABNORMALITIES = {"both": "mass,calcification"}


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

        self.n_views = self._infer_n_views()

        self.binarizer_type = LabelBinarizer().fit(self.meta["classification"])

        self.meta["abnormality"] = self.meta["abnormality"].replace(MAP_ABNORMALITIES)
        self.binarizer_abnorm = MultiLabelBinarizer().fit(
            [set(ab.split(",")) for ab in self.meta["abnormality"]]
        )
        self.meta["abnormality"] = self.meta["abnormality"].apply(
            lambda s: set(s.split(","))
        )

        self.meta = self.meta.loc[self.meta.split == split].reset_index()

        self.root_image_path = root_image_path
        self.n_samples = len(self.meta)
        self.image_size = image_size

        self.return_orig_image = return_orig_image

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

    def _infer_n_views(self):
        files_per_breast = (
            self.meta[["breast_id", "filename"]].groupby("breast_id").count().filename
        )

        return max(files_per_breast)

    def _preprocess_image(self, image: np.ndarray, is_left: bool):
        """
        1. Clean background noise
        2. Vertical mirroring to make all breast located on left side of frame
        3. Crop height to breast
        4. Resize
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

    def __len__(self):
        return len(self.meta.breast_id.unique())

    def _process_one_image(self, meta):
        orig_image = self._read_and_resize_image(meta.image_path)[..., None]
        image = self._preprocess_image(orig_image.copy(), meta.side == "left")

        return {"image": image, "orig_image": orig_image}

    def __getitem__(self, i):

        meta = self.meta.loc[self.meta.breast_id == self.meta.breast_id.iloc[i]]

        meta = meta.copy()

        image_paths = [
            (
                Path(self.root_image_path)
                / r.serie_id
                / (r.filename.split(".")[0] + ".png")
            )
            for _, r in meta.iterrows()
        ]
        meta["image_path"] = image_paths

        res = [self._process_one_image(r) for _, r in meta.iterrows()]

        # edge case: some patients have a single view following
        # we choose to duplicate the image to match the number of views
        # available in dominant case
        if len(res) < self.n_views:
            res = [res[0] for _ in range(self.n_views)]

        image = np.concatenate([r["image"] for r in res], axis=-1)
        orig_image = np.concatenate([r["orig_image"] for r in res], axis=-1)

        meta = meta.drop(columns=["filename", "image_path"])
        meta = meta.iloc[0]

        res = {
            "image": image,
            "orig_image": orig_image,
            "meta": meta,
            "t_type": self.binarizer_type.transform([meta.classification]),
            "t_abnorm": self.binarizer_abnorm.transform([meta.abnormality]),
        }

        if self.return_orig_image:
            res.update({"orig_image": orig_image})

        return res

    @staticmethod
    def collate_fn(samples: list[dict]):
        """
        convert list of samples obtained through __getitem__ to
        pytorch tensors
        """

        images = torch.tensor(
            np.stack([np.moveaxis(s["image"], 2, 0) for s in samples])
        ).float()

        t_type = torch.tensor(np.concatenate([s["t_type"] for s in samples]))
        t_abnorm = torch.tensor(np.concatenate([s["t_abnorm"] for s in samples]))

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
        image_size=1024,
        return_orig_image=True,
    )

    to_plot = [50, 10]
    fig, axes = plt.subplots(nrows=len(to_plot), ncols=2)

    for row, s_idx in enumerate(to_plot):
        sample = dset[s_idx]
        print(sample["meta"])
        image = sample["image"]
        orig_image = sample["orig_image"]

        concat_image = np.concatenate([image[..., 0], image[..., 1]], axis=1)
        concat_orig_image = np.concatenate(
            [orig_image[..., 0], orig_image[..., 1]], axis=1
        )
        axes[row][0].title.set_text("Original")
        axes[row][0].imshow(concat_orig_image)
        axes[row][1].title.set_text("Preprocessed")
        axes[row][1].imshow(concat_image)
    plt.show()
