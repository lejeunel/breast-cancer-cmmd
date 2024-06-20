from dataclasses import dataclass
from pathlib import Path

import typer
from rich.pretty import pprint
from typing_extensions import Annotated

from breastclf.ml.test import test
from breastclf.ml.train import train


@dataclass
class Experiment:
    fusion_mode: str
    lfabnorm: float
    name: str
    is_best: bool


def run_experiments(
    best_only: Annotated[
        bool, typer.Option(help="Skip experimental models and only run best model")
    ] = False,
    cuda: Annotated[bool, typer.Option(help="Use GPU acceleration")] = False,
):
    experiments = [
        Experiment("output", 0, "no_abnorm_fusion_output", False),
        Experiment("mean-feats", 0, "no_abnorm_fusion_mean", False),
        Experiment("max-feats", 0, "no_abnorm_fusion_max", False),
        Experiment("concat-feats", 0, "no_abnorm_fusion_cat", False),
        Experiment("output", 1, "abnorm_fusion_output", False),
        Experiment("mean-feats", 1, "abnorm_fusion_mean", True),
        Experiment("max-feats", 1, "abnorm_fusion_max", False),
        Experiment("concat-feats", 1, "abnorm_fusion_cat", False),
    ]

    root_data = Path("data")
    root_runs = Path("runs")

    for exp in experiments:
        if best_only and not exp.is_best:
            continue

        print("running experiment")
        pprint(exp)

        if not (root_runs / exp.name).exists():
            train(
                meta_path=root_data / "meta-images-split.csv",
                image_root_path=root_data / "png",
                out_root_path=root_runs,
                exp_name=exp.name,
                append_datetime_to_exp=False,
                fusion=exp.fusion_mode,
                lfabnorm=exp.lfabnorm,
                cuda=cuda,
            )
        else:
            print(f"skipping already existing experiment at {root_runs / exp.name}")

        if not (root_runs / exp.name / "test-results.csv").exists():
            test(
                meta_path=root_data / "meta-images-split.csv",
                image_root_path=root_data / "png",
                run_path=root_runs / exp.name,
                cuda=cuda,
            )
        else:
            print(
                f"skipping already existing testing results at {root_runs / exp.name / 'test-results.csv'}"
            )


if __name__ == "__main__":
    typer.run(run_experiments)
