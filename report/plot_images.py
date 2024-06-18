from breastclf.ml.dataloader import BreastDataset
from pathlib import Path
import matplotlib.pyplot as plt


dset = BreastDataset(
    Path("../data/png"),
    Path("../data/meta-images-split.csv"),
    split="train",
    image_size=1024,
)

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 20))

for breast_idx, axes_row in zip([301, 24, 397, 1035], axes):
    sample = dset.from_breast_id(breast_idx)
    meta = sample["meta"].reset_index()
    for i, ax in enumerate(axes_row):
        ax.imshow(
            sample["images"][i],
            cmap="gray",
        )
        ax.title.set_text(f"{meta.iloc[i].classification} / {meta.iloc[i].abnormality}")
        ax.set_xticks([])
        ax.set_yticks([])
plt.savefig("images/previews.png", bbox_inches="tight")
