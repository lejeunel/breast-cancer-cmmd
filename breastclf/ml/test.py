from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import typer
import yaml
from breastclf.ml.dataloader import make_dataloaders
from breastclf.ml.model import BreastClassifier
from rich.pretty import pprint
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
)
from tqdm import tqdm
from typing_extensions import Annotated


def test(
    meta_path: Annotated[Path, typer.Argument(help="path to meta-data csv file")],
    image_root_path: Annotated[Path, typer.Argument(help="root path to image files")],
    run_path: Annotated[Path, typer.Argument(help="root path")],
    n_workers: Annotated[
        int, typer.Option(help="Num of parallel processes in data loader")
    ] = 8,
    cuda: Annotated[Optional[bool], typer.Option(help="use GPU")] = True,
):
    """
    Testing routine.
    """

    cfg_path = run_path / "cfg.yaml"
    assert cfg_path.exists(), f"could not find run config at {cfg_path}"

    with open(run_path / "cfg.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    pprint("config:")
    pprint(cfg, expand_all=True)

    assert meta_path.exists(), f"could not find meta-data file at {meta_path}"

    dloader = make_dataloaders(
        image_root_path,
        meta_path,
        meta_backup_path=run_path,
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        n_workers=n_workers,
    )["test"]

    device = torch.device("cuda") if cuda else torch.device("cpu")
    model = BreastClassifier(fusion_mode=cfg["fusion"]).to(device)

    cp_root_path = run_path / "checkpoints"
    archive = torch.load(cp_root_path / "best.pth.tar")

    print("loading weights from checkpoint")
    model.load_state_dict(archive["model_state_dict"])

    model.eval()
    meta = []
    with torch.no_grad():
        for batch in (pbar := tqdm(dloader)):

            batch.to(device)
            batch, _ = model(batch)

            meta_ = batch.meta
            meta_["pred_type"] = batch.pred_type.cpu().numpy()
            meta_["tgt_type"] = batch.tgt_type.cpu().numpy()
            meta.append(meta_)

            pbar.set_description("[test]")

    out_path = run_path / "test-results.csv"
    meta = pd.concat(meta)

    meta = meta.drop_duplicates("breast_id")
    meta.drop(columns=["filename", "image_path", "index", "level_0"], inplace=True)

    print(f"saving breast-wise results to {out_path}")
    meta.to_csv(out_path, index=False)

    agg_results = pd.Series()
    for field_name, metric_fn, thr in [
        ("AUC(ROC)", roc_auc_score, None),
        ("F1", f1_score, 0.5),
        ("AP", average_precision_score, None),
    ]:
        y_true = meta["tgt_type"]
        y_pred = meta["pred_type"]
        if thr:
            y_pred = y_pred >= thr
        agg_results[field_name] = metric_fn(y_true, y_pred)

    agg_results["fusion_mode"] = cfg["fusion"]
    agg_results["lfabnorm"] = cfg["lfabnorm"] > 0
    agg_results["lftype"] = cfg["lftype"] > 0

    out_path = run_path / "test-agg-results.csv"
    print(f"saving aggregate performance results to {out_path}")
    pd.DataFrame(agg_results).T.to_csv(out_path, index=False)
