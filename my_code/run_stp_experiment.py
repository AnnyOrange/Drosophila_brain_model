from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd
from brian2 import Hz, ms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "my_code"))

from model import default_params, poi, construct_dataframe  # noqa: E402
import utils as utl  # noqa: E402
from sugar_circuit import SUGAR_NEURONS  # noqa: E402
from bitter_circuit import BITTER_NEURONS  # noqa: E402
from model_stp import create_model_stp  # noqa: E402


MN9_ID = 720575940660219265
CB0248_IDS = {720575940622695448, 720575940627383685}
CB0192_IDS = {720575940629888530}


def build_net_with_inputs(path_comp, path_con, params, stp_pre_ids, stp_post_ids, sugar_rate, bitter_rate):
    net, neu, syn_base, syn_stp, spk_mon, flyid2i = create_model_stp(path_comp, path_con, params, stp_pre_ids, stp_post_ids)
    inputs = []
    if sugar_rate > 0:
        exc = [flyid2i[f] for f in SUGAR_NEURONS if f in flyid2i]
        poi_sugar, neu = poi(neu, exc, [], params | {"r_poi": sugar_rate * Hz})
        inputs.extend(poi_sugar)
    if bitter_rate > 0:
        bitter_ids = [f for f in BITTER_NEURONS if f in flyid2i]
        poi_bitter, neu = poi(neu, bitter_ids, [], params | {"r_poi": bitter_rate * Hz})
        inputs.extend(poi_bitter)
    net.add(*inputs)
    return net, spk_mon, neu


def run_trials(n_run, path_comp, path_con, params, sugar_rate, bitter_rate, stp_pre_ids, stp_post_ids, exp_name, path_res):
    path_res.mkdir(parents=True, exist_ok=True)
    all_spk = []
    for _ in range(n_run):
        net, spk_mon, _ = build_net_with_inputs(path_comp, path_con, params, stp_pre_ids, stp_post_ids, sugar_rate, bitter_rate)
        net.run(params["t_run"])
        spk_dict = {k: v for k, v in spk_mon.spike_trains().items() if len(v)}
        all_spk.append(spk_dict)
    # map indices to flywire ids
    df_comp = pd.read_csv(path_comp, index_col=0)
    i2fid = {i: fid for i, fid in enumerate(df_comp.index)}
    df = construct_dataframe(all_spk, exp_name=exp_name, i2flyid=i2fid)
    out = path_res / f"{exp_name}.parquet"
    df.to_parquet(out, compression="brotli")
    return out


def main():
    ap = argparse.ArgumentParser(description="Run STP-limited sugar/bitter experiment toward MN9")
    ap.add_argument("--sugar-rate", type=float, default=150, help="Sugar Poisson Hz (0 to disable)")
    ap.add_argument("--bitter-rate", type=float, default=0, help="Bitter Poisson Hz (0 to disable)")
    ap.add_argument("--n-run", type=int, default=2)
    ap.add_argument("--path-comp", default="2023_03_23_completeness_630_final.csv")
    ap.add_argument("--path-con", default="2023_03_23_connectivity_630_final.parquet")
    ap.add_argument("--path-res", default="results/my_code_stp")
    ap.add_argument("--t-run-ms", type=float, default=1000)
    ap.add_argument("--exp-name", default="stp_test")
    ap.add_argument("--stp-U", type=float, default=0.5)
    ap.add_argument("--stp-tau-d-ms", type=float, default=200)
    ap.add_argument("--stp-tau-f-ms", type=float, default=500)
    ap.add_argument("--no-bg", action="store_true", help="Disable background Poisson noise")
    args = ap.parse_args()

    params = deepcopy(default_params)
    params["t_run"] = args.t_run_ms * ms
    params["n_run"] = args.n_run
    if args.no_bg:
        params["use_bg"] = False
    params["stp_U"] = args.stp_U
    params["stp_tau_d"] = args.stp_tau_d_ms * ms
    params["stp_tau_f"] = args.stp_tau_f_ms * ms

    stp_pre = set(SUGAR_NEURONS) | set(BITTER_NEURONS) | CB0248_IDS | CB0192_IDS
    stp_post = CB0248_IDS | CB0192_IDS | {MN9_ID}

    out = run_trials(
        n_run=args.n_run,
        path_comp=args.path_comp,
        path_con=args.path_con,
        params=params,
        sugar_rate=args.sugar_rate,
        bitter_rate=args.bitter_rate,
        stp_pre_ids=stp_pre,
        stp_post_ids=stp_post,
        exp_name=args.exp_name,
        path_res=Path(args.path_res),
    )
    df = utl.load_exps([str(out)])
    counts = df[df["flywire_id"] == MN9_ID].groupby("trial").size()
    print("Output:", out)
    print("MN9 per trial:", list(counts), "mean", counts.mean() if len(counts) else 0)


if __name__ == "__main__":
    main()
