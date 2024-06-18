from hmtest.ml.train import train
from hmtest.ml.test import test
import pandas as pd
from pathlib import Path
from rich.pretty import pprint

if __name__ == "__main__":

    experiments = [
        ("output", 0, "no_abnorm_fusion_output"),
        ("mean-feats", 0, "no_abnorm_fusion_mean"),
        ("max-feats", 0, "no_abnorm_fusion_max"),
        ("concat-feats", 0, "no_abnorm_fusion_cat"),
        ("output", 1, "abnorm_fusion_output"),
        ("mean-feats", 1, "abnorm_fusion_mean"),
        ("max-feats", 1, "abnorm_fusion_max"),
        ("concat-feats", 1, "abnorm_fusion_cat"),
    ]

    df = pd.DataFrame.from_records(
        [
            {k: exp[i] for i, k in enumerate(["fusion_mode", "lfabnorm", "exp_name"])}
            for exp in experiments
        ]
    )

    root_data = Path("data")
    root_runs = Path("runs")

    for _, exp in df.iterrows():
        print("running experiment")
        pprint(exp)

        if not (root_runs / exp.exp_name).exists():
            train(
                meta_path=root_data / "meta-images-split.csv",
                image_root_path=root_data / "png",
                out_root_path=root_runs,
                exp_name=exp.exp_name,
                append_datetime_to_exp=False,
                fusion=exp.fusion_mode,
                lfabnorm=exp.lfabnorm,
            )
        else:
            print(f"skipping already existing experiment at {root_runs / exp.exp_name}")

        if not (root_runs / exp.exp_name / "test-results.csv").exists():
            test(
                meta_path=root_data / "meta-images-split.csv",
                image_root_path=root_data / "png",
                run_path=root_runs / exp.exp_name,
            )
        else:
            print(
                f"skipping already existing testing results at {root_runs / exp.exp_name / 'test-results.csv'}"
            )
