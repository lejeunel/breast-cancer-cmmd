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
from breastclf.ml import image_preprocessing as impp


@dataclass
class Batch:
    """Container of input/outputs that
    we pass into trainer and callbacks"""

    images: list[np.array]
    meta: pd.DataFrame

    iter: int = 0

    pred_type: Optional[list[float]] = None
    pred_abnorm: Optional[list[list[float]]] = None

    tgt_type: Optional[list[int]] = None
    tgt_abnorm: Optional[list[list[int]]] = None

    loss_abnorm: Optional[float] = None
    loss_type: Optional[float] = None

    def set_loss(self, losses: torch.Tensor, field: str):
        assert field in ["type", "abnorm"], f"could not set loss for type {field}"
        setattr(self, "loss" + "_" + field, losses)

    def set_predictions(self, predictions: torch.Tensor, field: str):
        assert field in ["type", "abnorm"], f"could not set prediction for type {field}"
        setattr(self, "pred" + "_" + field, predictions)

    def to(self, device):
        """
        Send tensor to device
        """
        for attr in ["tgt_type", "tgt_abnorm", "images"]:
            setattr(self, attr, getattr(self, attr).to(device))

    def groupby(self, field):
        assert field in self.meta, f"groupby criteria {field} not found in meta fields"

        for _, g in self.meta.groupby(field, sort=False):
            b = Batch(
                meta=g,
                images=torch.cat([self.images[i][None, ...] for i in g.index]),
                tgt_type=torch.stack([self.tgt_type[i] for i in g.index]),
                tgt_abnorm=torch.stack([self.tgt_abnorm[i] for i in g.index]),
            )
            yield b

    def get_num_of_views(self):
        return self.meta.shape[0]

    @classmethod
    def from_list(cls, batches):
        return cls(
            meta=pd.concat([b.meta for b in batches]),
            images=torch.cat([b.images for b in batches]),
            iter=batches[0].iter,
            pred_type=torch.cat([b.pred_type for b in batches]),
            pred_abnorm=torch.cat([b.pred_abnorm for b in batches]),
            tgt_type=torch.cat([b.tgt_type for b in batches]),
            tgt_abnorm=torch.cat([b.tgt_abnorm for b in batches]),
        )


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
        with_pairs=False,
    ):
        """ """

        super().__init__()
        self.meta = pd.read_csv(meta_path)

        self.n_views = self._infer_n_views()
        self.with_pairs = with_pairs

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

        image = image / 255

        # if meta.classification == "Malignant":
        #     image = np.ones_like(image)
        # else:
        #     image = np.zeros_like(image)

        return {"image": image, "orig_image": orig_image}

    def from_breast_id(self, breast_id):
        meta = self.meta.loc[self.meta.breast_id == breast_id]
        return self._make_sample(meta)

    def _make_sample(self, meta):
        # edge case: some patients have a single view
        # we choose to duplicate the image to match the number of views
        # available in dominant case
        if meta.shape[0] < self.n_views:
            meta = pd.concat(self.n_views * [meta])

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

        images = np.stack([r["image"] for r in res])
        orig_images = np.stack([r["orig_image"] for r in res])

        res = {
            "images": images,
            "meta": meta,
            "t_type": self.binarizer_type.transform(meta.classification),
            "t_abnorm": self.binarizer_abnorm.transform(meta.abnormality),
        }

        if self.return_orig_image:
            res.update({"orig_images": orig_images})

        return res

    def __getitem__(self, i):

        meta = self.meta.loc[self.meta.breast_id == self.meta.breast_id.unique()[i]]

        return self._make_sample(meta)

    @staticmethod
    def collate_fn(samples: list[dict]):
        """
        Convert list of samples obtained through __getitem__ to
        pytorch tensors
        """

        images = torch.tensor(np.concatenate([s["images"] for s in samples])).float()
        images = torch.moveaxis(images, -1, 1)

        t_type = torch.tensor(np.concatenate([s["t_type"] for s in samples])).float()
        t_abnorm = torch.tensor(
            np.concatenate([s["t_abnorm"] for s in samples])
        ).float()

        meta = pd.concat([s["meta"] for s in samples]).reset_index()

        return Batch(
            **{
                "meta": meta,
                "images": images,
                "tgt_type": t_type,
                "tgt_abnorm": t_abnorm,
            }
        )


def make_dataloaders(
    image_root_path: Path,
    meta_path: Path,
    meta_backup_path: Path,
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

    print(f"saving meta-data files to {meta_backup_path}")
    for split, dset in datasets.items():
        dset.meta.to_csv(meta_backup_path / f"meta-{split}.csv", index=False)

    effective_batch_size = batch_size // datasets["train"].n_views

    dataloaders = {
        s: DataLoader(
            d,
            batch_size=effective_batch_size,
            num_workers=n_workers,
            collate_fn=BreastDataset.collate_fn,
            sampler=(
                RandomSampler(d, num_samples=n_batches_per_epoch * effective_batch_size)
                if s == "train"
                else None
            ),
        )
        for s, d in datasets.items()
    }

    return dataloaders
