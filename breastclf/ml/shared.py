from dataclasses import dataclass
import pandas as pd
import torch
from typing import Optional
import numpy as np


@dataclass
class Batch:
    """Container of input/outputs that
    we pass into trainer and callbacks"""

    ATTRIBUTES_TARGETS = ["tgt_type", "tgt_abnorm_calcification", "tgt_abnorm_mass"]
    ATTRIBUTES_PREDICTIONS = [
        "pred_type",
        "pred_abnorm_calcification",
        "pred_abnorm_mass",
    ]

    ATTRIBUTES_LOSSES = ["loss_abnorm", "loss_type"]

    meta: pd.DataFrame

    images: Optional[list[np.array]] = None

    iter: int = 0

    pred_type: Optional[list[float]] = None
    pred_abnorm_calcification: Optional[list[list[float]]] = None
    pred_abnorm_mass: Optional[list[list[float]]] = None

    tgt_type: Optional[list[int]] = None
    tgt_abnorm_calcification: Optional[list[list[int]]] = None
    tgt_abnorm_mass: Optional[list[list[int]]] = None

    loss_abnorm: Optional[float] = None
    loss_type: Optional[float] = None

    def set_loss(self, losses: torch.Tensor, field: str):
        assert field in ["type", "abnorm"], f"could not set loss for type {field}"
        setattr(self, "loss" + "_" + field, losses)

    def set_pred_type(self, predictions: torch.Tensor):
        self.pred_type = predictions

    def get_tgt_abnorm(self, subtype: str):
        assert subtype in [
            "mass",
            "calcification",
        ], f"no abnorm target with subtype {subtype}"
        return getattr(self, f"tgt_abnorm_{subtype}")

    def set_pred_abnorm(self, calcification: torch.Tensor, mass: torch.Tensor):
        self.pred_abnorm_calcification = calcification
        self.pred_abnorm_mass = mass

    def to(self, device):
        """
        Send all tensors to device
        """
        for attr in self.ATTRIBUTES_PREDICTIONS + self.ATTRIBUTES_TARGETS + ["images"]:
            current = getattr(self, attr)
            if current is not None:
                setattr(self, attr, current.to(device))

    def groupby(self, field):
        """
        Slice current batch with all values in column "field"
        Return a new Batch object
        """
        assert field in self.meta, f"groupby criteria {field} not found in meta fields"

        for _, g in self.meta.groupby(field, sort=False):

            b = Batch(
                meta=g, images=torch.cat([self.images[i][None, ...] for i in g.index])
            )
            for attr in self.ATTRIBUTES_PREDICTIONS + self.ATTRIBUTES_TARGETS:
                current = getattr(self, attr)
                if current is not None:
                    setattr(b, attr, torch.stack([current[i] for i in g.index]))

            yield b

    def get_num_of_views(self):
        return self.meta.shape[0]

    @classmethod
    def from_list(cls, batches):
        """
        Builds a batch from a list of batches
        """
        out = cls(
            meta=pd.concat([b.meta for b in batches]),
            iter=batches[0].iter,
        )
        for attr in cls.ATTRIBUTES_PREDICTIONS + cls.ATTRIBUTES_TARGETS + ["images"]:
            setattr(out, attr, torch.cat([getattr(b, attr) for b in batches]))
        return out
