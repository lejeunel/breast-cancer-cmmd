from pathlib import Path

import numpy as np
import pandas as pd
import torch
from breastclf.ml import image_preprocessing as impp
from breastclf.ml.shared import Batch
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder
from torch.utils.data import DataLoader, Dataset, RandomSampler


class AbnormalityEncoder:
    """
    Assigns to each abnormality an integer code such that:
    - -1 -> Undefined
    - 0 -> Absent
    - 1 -> Benign
    - 2 -> Malignant
    """

    def __init__(
        self,
        main_field="abnormality",
        secondary_field="classification",
        undefined_value="both",
    ):
        self.main_field = main_field
        self.secondary_field = secondary_field
        self.undefined_value = undefined_value
        self._encoders = {}

    def _set_abnormality_subtype(self, abnormality_type, meta: pd.DataFrame):
        """
        Sets a new temporary column 'ab_subtype' with the 'Absent' value
        using the 'secondary_field'

        Returns a copy of original dataframe.
        """
        meta_ = meta.copy()
        meta_["_ab_subtype"] = meta_[self.secondary_field]

        meta_.loc[meta_[self.main_field] != abnormality_type, "_ab_subtype"] = "Absent"
        meta_.loc[meta_[self.main_field] == self.undefined_value, "_ab_subtype"] = (
            np.nan
        )

        return meta_

    def fit(self, meta: pd.DataFrame):
        """
        Fit a binarizer on each abnormality
        """
        assert (
            self.undefined_value in meta[self.main_field].values
        ), f"requested undefined_value {self.undefined_value} not found in column {self.main_field}"

        for ab in meta[self.main_field].unique():
            if ab == self.undefined_value:
                continue
            meta_ = self._set_abnormality_subtype(ab, meta)
            X = meta_._ab_subtype.sort_values().unique().reshape(-1, 1)
            self._encoders[ab] = OrdinalEncoder().fit(X)

        return self

    def transform(self, meta: pd.DataFrame):
        """
        Returns for each binarizer a list of integers
        representing encodings
        """
        out = {}
        for ab, encoder in self._encoders.items():
            meta_ = self._set_abnormality_subtype(ab, meta)
            out[ab] = encoder.transform(meta_._ab_subtype.values.reshape(-1, 1))
            out[ab] = np.nan_to_num(out[ab], nan=-1)

        return out


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
        self.encoder_abnorm = AbnormalityEncoder().fit(self.meta)

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
        }

        # append abnormality targets with a prefix
        for k, v in self.encoder_abnorm.transform(meta).items():
            res.update({"t_abnorm_" + k: v})

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

        t_abnorm_calcification = torch.tensor(
            np.concatenate([s["t_abnorm_calcification"] for s in samples])
        ).long()
        t_abnorm_mass = torch.tensor(
            np.concatenate([s["t_abnorm_mass"] for s in samples])
        ).long()

        meta = pd.concat([s["meta"] for s in samples]).reset_index()

        return Batch(
            **{
                "meta": meta,
                "images": images,
                "tgt_type": t_type,
                "tgt_abnorm_calcification": t_abnorm_calcification,
                "tgt_abnorm_mass": t_abnorm_mass,
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


if __name__ == "__main__":
    root_path = Path("data")
    torch.manual_seed(4)
    dloaders = make_dataloaders(
        root_path / "png",
        root_path / "meta-images-split.csv",
        Path("/tmp"),
        batch_size=8,
        image_size=512,
        n_workers=8,
    )
    tgt_type = []
    classification = []
    for i, b in enumerate(dloaders["train"]):
        print(f"{i+1}/{len(dloaders['train'])}")
        tgt_type.append(b.tgt_type)
        classification.append(b.meta.classification.values)

    tgt_type = np.array([t_.numpy() for t in torch.cat(tgt_type) for t_ in t])
    classification = np.array([c_ for c in classification for c_ in c])
    all = np.vstack((tgt_type, classification)).T
    print(np.unique(all, axis=0))
