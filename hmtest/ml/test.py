from pathlib import Path
from typing import Optional

import torch
import typer
import yaml
from hmtest.ml.dataloader import BreastDataset
from hmtest.ml.model import BreastClassifier
from torch.utils.data import DataLoader
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
    batch_size: Annotated[int, typer.Option()] = 16,
):
    """
    Testing routine.
    """

    cfg_path = run_path / "cfg.yaml"
    assert cfg_path.exists(), f"could not find run config at {cfg_path}"

    with open(run_path / "cfg.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    print(f"config: {cfg}")

    assert meta_path.exists(), f"could not find meta-data file at {meta_path}"

    dset = BreastDataset(
        image_root_path,
        meta_path,
        split="test",
        image_size=cfg["image_size"],
    )

    effective_batch_size = batch_size // dset.n_views
    dloader = DataLoader(
        dset,
        batch_size=effective_batch_size,
        num_workers=n_workers,
        collate_fn=BreastDataset.collate_fn,
    )

    device = torch.device("cuda") if cuda else torch.device("cpu")
    model = BreastClassifier(fusion_mode=cfg["fusion"]).to(device)

    cp_root_path = run_path / "checkpoints"
    print(f"parsing {cp_root_path} for best model")
    cp_paths = sorted([f for f in cp_root_path.glob("*.pth.tar")])
    models = []
    for path in cp_paths:
        archive = torch.load(path)
        score = archive["metric"]
        models.append({"score": score, "path": path})

    print(models)
    best_model = max(models, key=lambda m: m["score"])
    print(f"found best model: {best_model['path']} with score {best_model['score']}")

    model.eval()
    with torch.no_grad():
        for batch in (pbar := tqdm(dloader)):

            batch.to(device)
            batch = model(batch)

            pbar.set_description("[test]")
