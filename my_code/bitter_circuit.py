from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
from brian2 import Hz

# ensure repo root import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import default_params, run_exp  # noqa: E402
import utils as utl  # noqa: E402


# Bitter candidate GRNs (labellar, MxLbN) from annotations: LB2* and LB4a
BITTER_NEURONS: List[int] = [
    720575940614273292, 720575940632293346, 720575940620856046, 720575940614734186,
    720575940622371037, 720575940619387814, 720575940627821896, 720575940637580102,
    720575940628832256, 720575940624604560, 720575940620564477, 720575940619663430,
    720575940615260378, 720575940611668066, 720575940637780970, 720575940613249959,
    720575940632706916, 720575940610465458, 720575940639395908, 720575940620966953,
    720575940631820225, 720575940624222209, 720575940633572167, 720575940625773112,
    720575940630413010, 720575940617896422,
]


@dataclass
class Paths:
    path_res: Path
    path_comp: Path
    path_con: Path
    n_proc: int = -1


def _mk_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_bitter(
    freqs: Sequence[int],
    paths: Paths,
    neu_ids: Sequence[int] = BITTER_NEURONS,
    force_overwrite: bool = False,
    base_params: dict | None = None,
) -> None:
    params = deepcopy(base_params or default_params)
    _mk_outdir(paths.path_res)
    # filter to IDs present in completeness file
    comp_ids = set(pd.read_csv(paths.path_comp, index_col=0).index)
    neu_present = [n for n in neu_ids if n in comp_ids]
    missing = [n for n in neu_ids if n not in comp_ids]
    if missing:
        print(f"Skipping {len(missing)} IDs not in completeness: {missing}")
    if not neu_present:
        raise SystemExit("No bitter neurons present in completeness file.")
    for f in freqs:
        params["r_poi"] = f * Hz
        run_exp(
            exp_name=f"bitterR_{f}Hz",
            neu_exc=list(neu_present),
            params=params,
            path_res=str(paths.path_res),
            path_comp=str(paths.path_comp),
            path_con=str(paths.path_con),
            n_proc=paths.n_proc,
            force_overwrite=force_overwrite,
        )


def collect_rates(parquet_files: Iterable[Path], t_run, n_run: int, flyid2name=None):
    df = utl.load_exps([str(p) for p in parquet_files])
    return utl.get_rate(df, t_run=t_run, n_run=n_run, flyid2name=flyid2name or {})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bitter circuit test")
    p.add_argument("--path-res", default="results/my_code_bitter", help="Output dir")
    p.add_argument("--path-comp", default="2023_03_23_completeness_630_final.csv")
    p.add_argument("--path-con", default="2023_03_23_connectivity_630_final.parquet")
    p.add_argument("--n-proc", type=int, default=1)
    p.add_argument("--n-run", type=int, default=None, help="Override n_run (default uses model.default_params)")
    p.add_argument("--freqs", type=int, nargs="+", default=[100])
    p.add_argument("--force-overwrite", action="store_true")
    p.add_argument("--summary", action="store_true", help="Print top responders and MN9 rate after run")
    p.add_argument("--no-bg", action="store_true", help="Disable background Poisson noise")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_params = deepcopy(default_params)
    if args.n_run is not None:
        base_params["n_run"] = args.n_run
    if args.no_bg:
        base_params["use_bg"] = False

    paths = Paths(
        path_res=Path(args.path_res),
        path_comp=Path(args.path_comp),
        path_con=Path(args.path_con),
        n_proc=args.n_proc,
    )
    run_bitter(
        freqs=args.freqs,
        paths=paths,
        base_params=base_params,
        force_overwrite=args.force_overwrite,
    )
    if args.summary:
        parquet_files = [paths.path_res / f"bitterR_{f}Hz.parquet" for f in args.freqs]
        df_rate, _ = collect_rates(parquet_files, t_run=base_params["t_run"], n_run=base_params["n_run"])
        col = df_rate.columns[0]
        print("Top 10 responders:")
        print(df_rate.sort_values(col, ascending=False).head(10))
        mn9 = 720575940660219265
        if mn9 in df_rate.index:
            print(f"MN9 rate: {df_rate.loc[mn9, col]} Hz")
        else:
            print("MN9 not present in results.")


if __name__ == "__main__":
    main()
