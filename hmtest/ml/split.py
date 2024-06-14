from pathlib import Path

import pandas as pd
import typer
from sklearn.model_selection import StratifiedShuffleSplit
from typing_extensions import Annotated


def _add_breast_id(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a column in the meta-data dataframe with a breast-id
    (combination of patient-id and breast side)
    """
    meta["breast_id"] = ""
    comb_idx = meta.groupby(["patient_id", "side"]).first().index
    for idx, comb in enumerate(comb_idx.values):
        same_breast_entries = (meta[["patient_id", "side"]] == comb).all(axis=1)
        meta.loc[same_breast_entries, "breast_id"] = idx

    return meta


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

    meta = pd.read_csv(meta_in)
    stratif_cols = stratif_cols.split(",")

    missing_cols = []
    for c in stratif_cols:
        if c not in meta.columns:
            missing_cols.append(c)

    assert (
        len(missing_cols) == 0
    ), f"provided non-existing columns for stratification: {missing_cols}"

    non_annotated = meta[stratif_cols].isna().sum(axis=1) == len(stratif_cols)
    print(f"removing {non_annotated.sum()} rows without values for {stratif_cols}")
    meta = meta.loc[~non_annotated].reset_index(drop=True)

    meta = _add_breast_id(meta)
    # validate that all images of the same breast have the same labels
    # meta.groupby("breast_id")[stratif_cols].nunique()

    breast_label_map = meta.groupby("breast_id")[stratif_cols].first()
    strat_label = pd.factorize(breast_label_map.agg("_".join, axis=1))[0]

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=int(test_size * len(breast_label_map)), random_state=seed
    )
    train_val_idx, test_idx = next(splitter.split(breast_label_map.index, strat_label))

    strat_label_train_val = strat_label[train_val_idx]
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=int(val_size * len(breast_label_map)), random_state=seed
    )
    train_idx, val_idx = next(splitter.split(train_val_idx, strat_label_train_val))
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    # add and fill new split column to dataframe
    meta["split"] = ""
    meta.loc[meta.breast_id.isin(train_idx), "split"] = "train"
    meta.loc[meta.breast_id.isin(val_idx), "split"] = "val"
    meta.loc[meta.breast_id.isin(test_idx), "split"] = "test"

    for split, g in meta.groupby("split"):
        if split:
            print(f"--- {split} ---")
            print(g.groupby(stratif_cols).size())

    print(f"saving to {meta_out}")
    meta.to_csv(meta_out, index=False)
