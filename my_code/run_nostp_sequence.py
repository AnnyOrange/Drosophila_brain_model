from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd
from brian2 import Network, PoissonGroup, Synapses, ms, Hz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "my_code"))

from model import default_params, create_model, construct_dataframe  # noqa: E402
import utils as utl  # noqa: E402
from sugar_circuit import SUGAR_NEURONS  # noqa: E402
from bitter_circuit import BITTER_NEURONS  # noqa: E402

MN9_ID = 720575940660219265


def build_base(path_comp, path_con, params):
    neu, _, spk_mon, bg_noise = create_model(path_comp, path_con, params)
    df_con = pd.read_parquet(path_con)
    df_comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}

    syn = Synapses(neu, neu, "w: volt", on_pre="g_post += w", delay=params["t_dly"], name="base_syn")
    syn.connect(i=df_con["Presynaptic_Index"].values, j=df_con["Postsynaptic_Index"].values)
    syn.w = df_con["Excitatory x Connectivity"].values * params["w_syn"]

    sugar_idx = [flyid2i[f] for f in SUGAR_NEURONS if f in flyid2i]
    bitter_idx = [flyid2i[f] for f in BITTER_NEURONS if f in flyid2i]

    pg_s = PoissonGroup(len(sugar_idx), rates=0 * Hz)
    syn_s = Synapses(pg_s, neu, on_pre="v_post += w_poi", model="w_poi: volt", name="poi_syn_s")
    syn_s.connect(i=list(range(len(sugar_idx))), j=sugar_idx)
    syn_s.w_poi = params["w_syn"] * params["f_poi"]

    pg_b = PoissonGroup(len(bitter_idx), rates=0 * Hz)
    syn_b = Synapses(pg_b, neu, on_pre="v_post += w_poi", model="w_poi: volt", name="poi_syn_b")
    syn_b.connect(i=list(range(len(bitter_idx))), j=bitter_idx)
    syn_b.w_poi = params["w_syn"] * params["f_poi"]

    net = Network(neu, syn, spk_mon, pg_s, syn_s, pg_b, syn_b, bg_noise)
    return net, spk_mon, pg_s, pg_b, df_comp


def run_single(rate_s, rate_b, phase_ms, exp, idx, path_res, no_bg=False):
    params = deepcopy(default_params)
    params["t_run"] = phase_ms * ms
    if no_bg:
        params["use_bg"] = False
    net, spk, pg_s, pg_b, df_comp = build_base("2023_03_23_completeness_630_final.csv", "2023_03_23_connectivity_630_final.parquet", params)
    pg_s.rates = rate_s * Hz
    pg_b.rates = rate_b * Hz
    net.run(phase_ms * ms)
    spk_dict = {k: v for k, v in spk.spike_trains().items() if len(v)}
    df = construct_dataframe([spk_dict], f"{exp}_run{idx}", {j: f for j, f in enumerate(df_comp.index)})
    out = Path(path_res) / f"{exp}_run{idx}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="brotli")
    return out


def run_seq(freq1, freq2, phase_ms, phase1_input, phase2_input, exp, idx, path_res, no_bg=False):
    params = deepcopy(default_params)
    params["t_run"] = phase_ms * ms
    if no_bg:
        params["use_bg"] = False
    net, spk, pg_s, pg_b, df_comp = build_base("2023_03_23_completeness_630_final.csv", "2023_03_23_connectivity_630_final.parquet", params)
    pg_s.rates = freq1 * Hz if phase1_input == "sugar" else 0 * Hz
    pg_b.rates = freq1 * Hz if phase1_input == "bitter" else 0 * Hz
    net.run(phase_ms * ms)
    pg_s.rates = freq2 * Hz if phase2_input == "sugar" else 0 * Hz
    pg_b.rates = freq2 * Hz if phase2_input == "bitter" else 0 * Hz
    net.run(phase_ms * ms)
    spk_dict = {k: v for k, v in spk.spike_trains().items() if len(v)}
    df = construct_dataframe([spk_dict], f"{exp}_run{idx}", {j: f for j, f in enumerate(df_comp.index)})
    out = Path(path_res) / f"{exp}_run{idx}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="brotli")
    return out


def mn9_phase_counts(path, phase_ms):
    df = utl.load_exps([str(path)])
    mn9 = df[df["flywire_id"] == MN9_ID]
    p1 = len(mn9[mn9["t"] < phase_ms / 1000.0])
    p2 = len(mn9[(mn9["t"] >= phase_ms / 1000.0) & (mn9["t"] < 2 * phase_ms / 1000.0)])
    return p1, p2, len(mn9)


def main():
    ap = argparse.ArgumentParser(description="No-STP sequences for sugar/bitter")
    ap.add_argument("--n-run", type=int, default=20)
    ap.add_argument("--phase-ms", type=int, default=1000)
    ap.add_argument("--path-res", default="results/my_code_mixed")
    ap.add_argument("--no-bg", action="store_true", help="Disable background Poisson noise")
    args = ap.parse_args()

    # 47-only
    for i in range(args.n_run):
        run_single(47, 0, args.phase_ms, "nostp_sugar47", i, args.path_res, no_bg=args.no_bg)
    # 150->47 sugar
    for i in range(args.n_run):
        run_seq(150, 47, args.phase_ms, "sugar", "sugar", "nostp_sugar150_47", i, args.path_res, no_bg=args.no_bg)
    # 150 bitter -> 47 sugar
    for i in range(args.n_run):
        run_seq(150, 47, args.phase_ms, "bitter", "sugar", "nostp_bitter150_sugar47", i, args.path_res, no_bg=args.no_bg)

    # summary
    import glob
    paths = sorted(glob.glob(f"{args.path_res}/nostp_sugar47_run*.parquet"))
    counts = [mn9_phase_counts(p, args.phase_ms)[2] for p in paths]
    if counts:
        mean = sum(counts) / len(counts)
        print("47-only mean", mean)
    paths = sorted(glob.glob(f"{args.path_res}/nostp_sugar150_47_run*.parquet"))
    phase2 = [mn9_phase_counts(p, args.phase_ms)[1] for p in paths]
    if phase2:
        mean = sum(phase2) / len(phase2)
        print("150->47 sugar phase2 mean", mean)
    paths = sorted(glob.glob(f"{args.path_res}/nostp_bitter150_sugar47_run*.parquet"))
    phase2b = [mn9_phase_counts(p, args.phase_ms)[1] for p in paths]
    if phase2b:
        mean = sum(phase2b) / len(phase2b)
        print("bitter150->47 sugar phase2 mean", mean)


if __name__ == "__main__":
    main()
