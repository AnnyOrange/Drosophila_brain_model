"""
Run mixed/sequence stimuli in a single simulation without resetting the network.
Allows changing Poisson rates between phases to observe potential “memory” effects.
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Sequence

import pandas as pd
from brian2 import Network, SpikeMonitor, Hz, ms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "my_code"))

from model import default_params, create_model, poi  # noqa: E402
import utils as utl  # noqa: E402
from sugar_circuit import SUGAR_NEURONS  # noqa: E402
from bitter_circuit import BITTER_NEURONS  # noqa: E402

MN9_ID = 720575940660219265


def filtered_bitter(path_comp: Path) -> List[int]:
    comp_ids = set(pd.read_csv(path_comp, index_col=0).index)
    return [n for n in BITTER_NEURONS if n in comp_ids]


def run_sequence(
    path_comp: Path,
    path_con: Path,
    sugar_rates: Sequence[float],
    bitter_rates: Sequence[float],
    phase_duration_ms: float,
    path_res: Path,
    exp_name: str,
    no_bg: bool = False,
) -> tuple[Path, int]:
    params = deepcopy(default_params)
    params["t_run"] = phase_duration_ms * ms  # per phase duration
    if no_bg:
        params["use_bg"] = False
    # build model once
    neu, syn, spk_mon, bg_noise = create_model(str(path_comp), str(path_con), params)

    # map flywire IDs to brian indices
    df_comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}

    inputs = []
    poi_sugar = None
    poi_bitter = None
    if sugar_rates:
        exc_idx = [flyid2i[fid] for fid in SUGAR_NEURONS if fid in flyid2i]
        poi_sugar_list, neu = poi(neu, exc_idx, [], params)
        inputs.extend(poi_sugar_list)
        poi_sugar = poi_sugar_list  # list of PoissonInput
    if bitter_rates:
        bitter_ids = filtered_bitter(path_comp)
        exc_idx_bitter = [flyid2i[fid] for fid in bitter_ids if fid in flyid2i]
        poi_bitter_list, neu = poi(neu, exc_idx_bitter, [], params)
        inputs.extend(poi_bitter_list)
        poi_bitter = poi_bitter_list

    net = Network(neu, syn, spk_mon, bg_noise, *inputs)

    # run each phase, adjusting rates
    n_phase = max(len(sugar_rates), len(bitter_rates))
    for i in range(n_phase):
        if sugar_rates and i < len(sugar_rates) and poi_sugar:
            for p in poi_sugar:
                p.rate = sugar_rates[i] * Hz
        if bitter_rates and i < len(bitter_rates) and poi_bitter:
            for p in poi_bitter:
                p.rate = bitter_rates[i] * Hz
        net.run(params["t_run"])

    # save spikes
    spk_dict = {k: v for k, v in spk_mon.spike_trains().items() if len(v)}
    res = [spk_dict]  # single trial list for construct_dataframe
    comp_df = pd.read_csv(path_comp, index_col=0)
    i2flyid = {i: fid for i, fid in enumerate(comp_df.index)}
    df = utl.construct_dataframe(res=res, exp_name=exp_name, i2flyid=i2flyid)

    out_path = path_res / f"{exp_name}.parquet"
    path_res.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="brotli")
    return out_path, n_phase


def summarize_mn9(parquet_path: Path, phase_duration_ms: float, n_phase: int):
    df = utl.load_exps([str(parquet_path)])
    mn9 = df[df["flywire_id"] == MN9_ID]
    counts = mn9.groupby("trial").size()
    mean = counts.mean() if len(counts) else 0
    std = counts.std(ddof=0) if len(counts) > 1 else 0
    # per-phase counts based on spike times (seconds)
    per_phase = []
    for i in range(n_phase):
        t0 = (phase_duration_ms * i) / 1000.0
        t1 = (phase_duration_ms * (i + 1)) / 1000.0
        per_phase.append(len(mn9[(mn9["t"] >= t0) & (mn9["t"] < t1)]))
    return list(counts), mean, std, per_phase


def main():
    ap = argparse.ArgumentParser(description="Mixed stimulus (sequential rates in one simulation)")
    ap.add_argument("--sugar-rates", default="150,47", help="Comma-separated Hz for sugar phases; empty for none")
    ap.add_argument("--bitter-rates", default="", help="Comma-separated Hz for bitter phases; empty for none")
    ap.add_argument("--phase-ms", type=float, default=1000, help="Duration per phase in ms")
    ap.add_argument("--path-res", default="results/my_code_mixed", help="Output directory")
    ap.add_argument("--exp-name", default="mixed_seq", help="Output basename")
    ap.add_argument("--path-comp", default="2023_03_23_completeness_630_final.csv")
    ap.add_argument("--path-con", default="2023_03_23_connectivity_630_final.parquet")
    ap.add_argument("--no-bg", action="store_true", help="Disable background Poisson noise")
    args = ap.parse_args()

    sugar_rates = [float(x) for x in args.sugar_rates.split(",") if x.strip()] if args.sugar_rates else []
    bitter_rates = [float(x) for x in args.bitter_rates.split(",") if x.strip()] if args.bitter_rates else []

    out_path, n_phase = run_sequence(
        path_comp=Path(args.path_comp),
        path_con=Path(args.path_con),
        sugar_rates=sugar_rates,
        bitter_rates=bitter_rates,
        phase_duration_ms=args.phase_ms,
        path_res=Path(args.path_res),
        exp_name=args.exp_name,
        no_bg=args.no_bg,
    )
    counts, mean, std, per_phase = summarize_mn9(out_path, phase_duration_ms=args.phase_ms, n_phase=n_phase)
    print(f"Saved spikes to {out_path}")
    print(f"MN9 per-phase spikes: {counts}, mean {mean:.2f}, std {std:.2f}")
    print(f"Per-phase windows counts: {per_phase}")


if __name__ == "__main__":
    main()
