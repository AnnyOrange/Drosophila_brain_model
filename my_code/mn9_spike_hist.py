from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from brian2 import ms, Hz
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import default_params, run_exp  # noqa: E402
import utils as utl  # noqa: E402
from sugar_circuit import SUGAR_NEURONS  # noqa: E402
from bitter_circuit import BITTER_NEURONS  # noqa: E402


MN9_ID = 720575940660219265


def run_and_hist(exp_name: str, neu_exc, path_res: Path, path_comp: Path, path_con: Path, params: dict):
    path_res.mkdir(parents=True, exist_ok=True)
    # run experiment
    run_exp(
        exp_name=exp_name,
        neu_exc=list(neu_exc),
        params=params,
        path_res=str(path_res),
        path_comp=str(path_comp),
        path_con=str(path_con),
        n_proc=1,
        force_overwrite=True,
    )
    parquet_path = path_res / f"{exp_name}.parquet"
    df = utl.load_exps([str(parquet_path)])
    df_mn9 = df[df["flywire_id"] == MN9_ID]
    return df_mn9, parquet_path


def plot_hist(df_mn9: pd.DataFrame, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 3))
    if df_mn9.empty:
        plt.title(f"{title} (no spikes)")
        plt.xlabel("Time (s)")
        plt.ylabel("Spike count")
    else:
        plt.hist(df_mn9["t"], bins=50, range=(0, 1.0), color="tab:blue", edgecolor="black")
        plt.xlabel("Time (s)")
        plt.ylabel("Spike count")
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_trial_counts(df_mn9: pd.DataFrame, out_path: Path, title: str):
    """Bar + scatter of per-trial spike counts for MN9."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    if df_mn9.empty:
        plt.title(f"{title} (no spikes)")
        plt.xlabel("Trial")
        plt.ylabel("Spikes in 1 s")
    else:
        counts = df_mn9.groupby("trial").size()
        plt.bar(counts.index, counts.values, color="tab:orange", edgecolor="black")
        plt.scatter(counts.index, counts.values, color="black", s=20, zorder=3)
        plt.xlabel("Trial")
        plt.ylabel("Spikes in 1 s")
        plt.title(f"{title}\nmean={counts.mean():.2f}, std={counts.std():.2f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Run 1s stim and plot MN9 spikes")
    ap.add_argument("--mode", choices=["both", "sugar", "bitter"], default="both", help="Which experiment to run")
    ap.add_argument("--n-run", type=int, default=2, help="Number of trials")
    ap.add_argument("--r-poi", type=float, default=None, help="Override Poisson rate (Hz) for excitation")
    args = ap.parse_args()

    base_params = deepcopy(default_params)
    base_params["t_run"] = 1000 * ms
    base_params["n_run"] = args.n_run
    if args.r_poi is not None:
        base_params["r_poi"] = args.r_poi * Hz

    path_res = Path("results/my_code_hist")
    path_comp = Path("2023_03_23_completeness_630_final.csv")
    path_con = Path("2023_03_23_connectivity_630_final.parquet")

    # sugar
    if args.mode in ("both", "sugar"):
        df_sugar, pq_sugar = run_and_hist(
            exp_name="sugarR_hist",
            neu_exc=SUGAR_NEURONS,
            path_res=path_res,
            path_comp=path_comp,
            path_con=path_con,
            params=base_params,
        )
        plot_hist(df_sugar, path_res / "mn9_sugar_hist.png", "MN9 spikes with sugar GRN (1s)")
        # per-trial counts
        sugar_counts = df_sugar.groupby("trial").size()
        sugar_counts.to_csv(path_res / "mn9_sugar_trial_counts.csv", header=["spikes"])
        plot_trial_counts(df_sugar, path_res / "mn9_sugar_trial_counts.png", "MN9 spikes/trial with sugar GRN")
        print("Sugar MN9 spikes:", len(df_sugar), "parquet:", pq_sugar)
        print("Sugar per-trial mean:", sugar_counts.mean(), "std:", sugar_counts.std())

    # bitter (only IDs present in completeness)
    if args.mode in ("both", "bitter"):
        comp_ids = set(pd.read_csv(path_comp, index_col=0).index)
        bitter_ids = [n for n in BITTER_NEURONS if n in comp_ids]
        df_bitter, pq_bitter = run_and_hist(
            exp_name="bitterR_hist",
            neu_exc=bitter_ids,
            path_res=path_res,
            path_comp=path_comp,
            path_con=path_con,
            params=base_params,
        )
        plot_hist(df_bitter, path_res / "mn9_bitter_hist.png", "MN9 spikes with bitter GRN (1s)")
        bitter_counts = df_bitter.groupby("trial").size()
        bitter_counts.to_csv(path_res / "mn9_bitter_trial_counts.csv", header=["spikes"])
        plot_trial_counts(df_bitter, path_res / "mn9_bitter_trial_counts.png", "MN9 spikes/trial with bitter GRN")
        print("Bitter MN9 spikes:", len(df_bitter), "parquet:", pq_bitter)
        print("Bitter per-trial mean:", bitter_counts.mean(), "std:", bitter_counts.std())


if __name__ == "__main__":
    main()
