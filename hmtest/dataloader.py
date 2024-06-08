from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision.models.detection.image_list import ImageList

from obj_detection.bbox_utils import SquarePadder


class CarDataset(Dataset):
    """Loads raw-images and bbox coordinates to produce tensors."""

    def __init__(
        self,
        csv_file: Path,
        root_dir: Path,
        out_size: int = 512,
        split: str | None = None,
    ):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.meta = pd.read_csv(root_dir / csv_file)

        if split:
            self.meta = self.meta.loc[self.meta.split == split]

        self.root_dir = root_dir
        self.out_size = out_size

    def num_classes(self):
        return self.meta["class"].unique().size

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        im_path = self.root_dir / self.meta.iloc[idx].relative_im_path
        assert im_path.exists(), f"Could not resolve path {im_path}"

        image = io.imread(im_path)

        padder = SquarePadder()
        padder.fit(image.shape[:2])

        # pad image and bbox coordinates to make them squared and resize
        image = padder.transform_image(image)
        image = transform.resize(image, (self.out_size, self.out_size))

        bbox_coords_top = self.meta.iloc[idx][["bbox_y1", "bbox_x1"]]
        bbox_coords_top = padder.transform_coords(*bbox_coords_top.values)

        bbox_coords_bottom = self.meta.iloc[idx][["bbox_y2", "bbox_x2"]]
        bbox_coords_bottom = padder.transform_coords(
            *bbox_coords_bottom.values
        )

        bbox_coords_top = [
            int(c * self.out_size / padder.w) for c in bbox_coords_top
        ]
        bbox_coords_bottom = [
            int(c * self.out_size / padder.w) for c in bbox_coords_bottom
        ]

        return {
            "meta": self.meta.iloc[idx],
            "image": image,
            "bbox_xy": np.vstack(
                [bbox_coords_top[::-1], bbox_coords_bottom[::-1]]
            ),
            "class_id": self.meta.iloc[idx]["class"],
        }

    @staticmethod
    def collate_fn(samples: list[dict]):
        """
        convert list of samples obtained through __getitem__ to
        pytorch tensors
        """
        import torch

        images = torch.tensor(
            np.stack([np.moveaxis(s["image"], 2, 0) for s in samples])
        ).float()
        boxes = [torch.tensor(s["bbox_xy"]).reshape(1, -1) for s in samples]
        targets = torch.tensor(np.stack([s["class_id"] for s in samples]))

        return {
            "meta": pd.DataFrame([s["meta"] for s in samples]),
            "images": ImageList(
                tensors=images, image_sizes=[s["image"].shape for s in samples]
            ),
            "boxes": boxes,
            "targets": targets,
        }


if __name__ == "__main__":
    test = CarDataset()
