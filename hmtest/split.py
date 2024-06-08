from pathlib import Path
from typing_extensions import Annotated
import typer
import numpy as np


def split(
    meta_in: Path,
    meta_out: Path,
    val_size: float,
    test_size: float,
    stratif_cols: Annotated[
        str,
        typer.Option(help="comma-separated list of column names on which to stratify"),
    ] = "abnormality,classification",
    seed: Annotated[int, typer.Option(help="Random seed for shuffling")] = 42,
):
    """
    Takes an input annotated meta-data file (one row per image),
    and assigns each row to the train, val, or test set
    using their respective relative size and a random seed.
    """

    assert val_size + test_size < 1.0, f"val and test sizes must be < 1"

    from sklearn.model_selection import StratifiedShuffleSplit
    import pandas as pd

    meta = pd.read_csv(meta_in)
    stratif_cols = stratif_cols.split(",")

    missing_cols = []
    for c in stratif_cols:
        if c not in meta.columns:
            missing_cols.append(c)

    assert (
        len(missing_cols) == 0
    ), f"provided non-existing columns for stratification: {missing_cols}"

    strat_label = meta[stratif_cols].apply(
        lambda row: "_".join(row.values.astype(str)), axis=1
    )
    strat_label = pd.factorize(strat_label)[0]

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )

    train_val_idx, test_idx = next(
        splitter.split(X=np.zeros_like(strat_label), y=strat_label)
    )

    strat_label_train_val = strat_label[train_val_idx]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(train_val_idx, strat_label_train_val))

    meta["split"] = ""
    meta.loc[train_idx, "split"] = "train"
    meta.loc[val_idx, "split"] = "val"
    meta.loc[test_idx, "split"] = "test"

    for split, g in meta.groupby("split"):
        if split:
            print(f"--- {split} ---")
            print(g.groupby(stratif_cols).size())

    breakpoint()
