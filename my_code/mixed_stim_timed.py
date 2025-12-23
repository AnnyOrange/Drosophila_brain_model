"""
Mixed stimulus with two sequential phases in a single run (no network reset).
Phase 1: sugar at freq1 Hz for 1 s
Phase 2: sugar at freq2 Hz for 1 s
Outputs MN9 spikes and per-phase counts.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import argparse
from brian2 import Network, PoissonGroup, Synapses, ms, Hz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "my_code"))

from model import default_params, create_model, construct_dataframe  # noqa: E402
import utils as utl  # noqa: E402
from sugar_circuit import SUGAR_NEURONS  # noqa: E402


MN9_ID = 720575940660219265


def build_net(path_comp: str, path_con: str, params: dict, freq: float):
    """Build network once, connect sugar PoissonGroup with adjustable rate via Synapses."""
    # Load data
    df_comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}

    # Neurons
    neu, syn, spk_mon, bg_noise = create_model(path_comp, path_con, params)

    # Sugar indices
    exc_idx = [flyid2i[fid] for fid in SUGAR_NEURONS if fid in flyid2i]

    # Poisson group and synapses
    pg = PoissonGroup(len(exc_idx), rates=freq * Hz)
    syn_poi = Synapses(pg, neu, on_pre="v_post += w_poi", model="w_poi : volt", name="poi_syn")
    syn_poi.connect(i=list(range(len(exc_idx))), j=exc_idx)
    syn_poi.w_poi = params["w_syn"] * params["f_poi"]

    net = Network(neu, syn, syn_poi, pg, spk_mon, bg_noise)
    return net, pg, spk_mon, df_comp


def run_two_phase(freq1: float, freq2: float, path_comp: str, path_con: str, params: dict, phase_ms: int, exp_name: str, path_res: Path):
    net, pg, spk_mon, df_comp = build_net(path_comp, path_con, params, freq1)
    # Phase 1
    pg.rates = freq1 * Hz
    net.run(phase_ms * ms)
    # Phase 2
    pg.rates = freq2 * Hz
    net.run(phase_ms * ms)

    # save spikes
    spk_dict = {k: v for k, v in spk_mon.spike_trains().items() if len(v)}
    res = [spk_dict]
    i2flyid = {i: fid for i, fid in enumerate(df_comp.index)}
    df = construct_dataframe(res=res, exp_name=exp_name, i2flyid=i2flyid)
    out_path = path_res / f"{exp_name}.parquet"
    path_res.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="brotli")
    return out_path, phase_ms


def summarize(parquet_path: Path, phase_ms: int):
    df = utl.load_exps([str(parquet_path)])
    mn9 = df[df["flywire_id"] == MN9_ID]
    counts_total = len(mn9)
    phase1 = len(mn9[mn9["t"] < phase_ms / 1000.0])
    phase2 = len(mn9[(mn9["t"] >= phase_ms / 1000.0) & (mn9["t"] < 2 * phase_ms / 1000.0)])
    return counts_total, phase1, phase2


def main():
    parser = argparse.ArgumentParser(description="Two-phase sugar stimulation in one run")
    parser.add_argument("--freq1", type=float, default=150, help="Sugar rate phase 1 (Hz)")
    parser.add_argument("--freq2", type=float, default=47, help="Sugar rate phase 2 (Hz)")
    parser.add_argument("--phase-ms", type=int, default=1000, help="Duration per phase (ms)")
    parser.add_argument("--n-run", type=int, default=1, help="Number of sequences to run")
    parser.add_argument("--path-comp", default="2023_03_23_completeness_630_final.csv")
    parser.add_argument("--path-con", default="2023_03_23_connectivity_630_final.parquet")
    parser.add_argument("--path-res", default="results/my_code_mixed")
    parser.add_argument("--no-bg", action="store_true", help="Disable background Poisson noise")
    args = parser.parse_args()

    for run_idx in range(args.n_run):
        params = deepcopy(default_params)
        params["t_run"] = args.phase_ms * ms  # not used directly, but keep consistent
        params["n_run"] = 1
        if args.no_bg:
            params["use_bg"] = False
        exp_name = f"mixed_sugar_{int(args.freq1)}_{int(args.freq2)}_run{run_idx}"
        out_path, _ = run_two_phase(
            freq1=args.freq1,
            freq2=args.freq2,
            path_comp=args.path_comp,
            path_con=args.path_con,
            params=params,
            phase_ms=args.phase_ms,
            exp_name=exp_name,
            path_res=Path(args.path_res),
        )
        total, p1, p2 = summarize(out_path, args.phase_ms)
        print(f"[{run_idx}] Saved to {out_path} | MN9 total {total}, phase1 {p1}, phase2 {p2}")


if __name__ == "__main__":
    main()
