from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd
from brian2 import Network, PoissonGroup, Synapses, SpikeMonitor, ms, Hz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "my_code"))

from model import default_params, create_model, construct_dataframe  # noqa: E402
import utils as utl  # noqa: E402
from sugar_circuit import SUGAR_NEURONS  # noqa: E402
from bitter_circuit import BITTER_NEURONS  # noqa: E402
from water_circuit import WATER_NEURONS  # noqa: E402

MN9_ID = 720575940660219265
CB0248_IDS = {720575940622695448, 720575940627383685}
CB0192_IDS = {720575940629888530}


def build_net_with_stp(path_comp, path_con, params, stp_pre_ids, stp_post_ids, freq_sugar, freq_bitter, freq_water):
    # base neurons
    neu, _, spk_mon, bg_noise = create_model(path_comp, path_con, params)
    df_con = pd.read_parquet(path_con)
    df_comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}

    # split edges
    mask = df_con["Presynaptic_ID"].isin(stp_pre_ids) & df_con["Postsynaptic_ID"].isin(stp_post_ids)
    df_stp = df_con[mask]
    df_base = df_con[~mask]

    # base synapses
    syn_base = Synapses(neu, neu, "w : volt", on_pre="g_post += w", delay=params["t_dly"], name="base_synapses")
    syn_base.connect(i=df_base["Presynaptic_Index"].values, j=df_base["Postsynaptic_Index"].values)
    syn_base.w = df_base["Excitatory x Connectivity"].values * params["w_syn"]

    # STP synapses
    syn_stp = Synapses(
        neu,
        neu,
        """
        w_base : volt
        u : 1
        x : 1
        t_last : second
        """,
        on_pre="""
        u = U + (u - U)*exp(-(t - t_last)/tau_f)
        x = 1 - (1 - x)*exp(-(t - t_last)/tau_d)
        g_post += w_base * u * x
        x -= u * x
        t_last = t
        """,
        delay=params["t_dly"],
        method="euler",
        namespace={
            "U": params.get("stp_U", 0.5),
            "tau_d": params.get("stp_tau_d", 200 * ms),
            "tau_f": params.get("stp_tau_f", 500 * ms),
        },
        name="stp_synapses",
    )
    syn_stp.connect(i=df_stp["Presynaptic_Index"].values, j=df_stp["Postsynaptic_Index"].values)
    if len(df_stp):
        syn_stp.w_base = df_stp["Excitatory x Connectivity"].values * params["w_syn"]
        syn_stp.u = params.get("stp_u0", params.get("stp_U", 0.5))
        syn_stp.x = 1

    # Poisson inputs
    sugar_idx = [flyid2i[f] for f in SUGAR_NEURONS if f in flyid2i]
    bitter_ids = [f for f in BITTER_NEURONS if f in flyid2i]
    bitter_idx = [flyid2i[f] for f in bitter_ids]
    water_idx = [flyid2i[f] for f in WATER_NEURONS if f in flyid2i]

    pg_s = PoissonGroup(len(sugar_idx), rates=freq_sugar * Hz)
    syn_s = Synapses(pg_s, neu, on_pre="v_post += w_poi", model="w_poi : volt", name="poi_syn_s")
    syn_s.connect(i=list(range(len(sugar_idx))), j=sugar_idx)
    syn_s.w_poi = params["w_syn"] * params["f_poi"]

    pg_b = PoissonGroup(len(bitter_idx), rates=freq_bitter * Hz)
    syn_b = Synapses(pg_b, neu, on_pre="v_post += w_poi", model="w_poi : volt", name="poi_syn_b")
    syn_b.connect(i=list(range(len(bitter_idx))), j=bitter_idx)
    syn_b.w_poi = params["w_syn"] * params["f_poi"]

    pg_w = PoissonGroup(len(water_idx), rates=freq_water * Hz)
    syn_w = Synapses(pg_w, neu, on_pre="v_post += w_poi", model="w_poi : volt", name="poi_syn_w")
    syn_w.connect(i=list(range(len(water_idx))), j=water_idx)
    syn_w.w_poi = params["w_syn"] * params["f_poi"]

    net = Network(neu, syn_base, syn_stp, syn_s, syn_b, syn_w, pg_s, pg_b, pg_w, spk_mon, bg_noise)
    return net, pg_s, pg_b, pg_w, spk_mon, df_comp


def run_sequence(freq1, freq2, phase_ms, path_comp, path_con, params, stp_pre, stp_post, exp_name, path_res, phase1_input, phase2_input):
    # initialize with phase1 rates
    init_s = freq1 if phase1_input == "sugar" else (freq2 if phase2_input == "sugar" else 0)
    init_b = freq1 if phase1_input == "bitter" else (freq2 if phase2_input == "bitter" else 0)
    init_w = freq1 if phase1_input == "water" else (freq2 if phase2_input == "water" else 0)
    net, pg_s, pg_b, pg_w, spk_mon, df_comp = build_net_with_stp(
        path_comp, path_con, params, stp_pre, stp_post, init_s, init_b, init_w
    )

    # Phase 1
    pg_s.rates = (freq1 * Hz) if phase1_input == "sugar" else 0 * Hz
    pg_b.rates = (freq1 * Hz) if phase1_input == "bitter" else 0 * Hz
    pg_w.rates = (freq1 * Hz) if phase1_input == "water" else 0 * Hz
    net.run(phase_ms * ms)
    # Phase 2
    pg_s.rates = (freq2 * Hz) if phase2_input == "sugar" else 0 * Hz
    pg_b.rates = (freq2 * Hz) if phase2_input == "bitter" else 0 * Hz
    pg_w.rates = (freq2 * Hz) if phase2_input == "water" else 0 * Hz
    net.run(phase_ms * ms)

    spk_dict = {k: v for k, v in spk_mon.spike_trains().items() if len(v)}
    res = [spk_dict]
    i2fid = {i: fid for i, fid in enumerate(df_comp.index)}
    df = construct_dataframe(res, exp_name=exp_name, i2flyid=i2fid)
    out = Path(path_res) / f"{exp_name}.parquet"
    Path(path_res).mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="brotli")
    return out


def summarize(parquet_path, phase_ms):
    df = utl.load_exps([str(parquet_path)])
    mn9 = df[df["flywire_id"] == MN9_ID]
    total = len(mn9)
    p1 = len(mn9[mn9["t"] < phase_ms / 1000.0])
    p2 = len(mn9[(mn9["t"] >= phase_ms / 1000.0) & (mn9["t"] < 2 * phase_ms / 1000.0)])
    return total, p1, p2


def main():
    ap = argparse.ArgumentParser(description="Two-phase sugar with STP on key edges")
    ap.add_argument("--freq1", type=float, default=150)
    ap.add_argument("--freq2", type=float, default=47)
    ap.add_argument("--phase-ms", type=float, default=1000)
    ap.add_argument("--phase1-input", choices=["sugar", "bitter", "water"], default="sugar")
    ap.add_argument("--phase2-input", choices=["sugar", "bitter", "water"], default="sugar")
    ap.add_argument("--path-comp", default="2023_03_23_completeness_630_final.csv")
    ap.add_argument("--path-con", default="2023_03_23_connectivity_630_final.parquet")
    ap.add_argument("--path-res", default="results/my_code_mixed")
    ap.add_argument("--exp-name", default="mixed_stp")
    ap.add_argument("--n-run", type=int, default=1, help="Number of sequences (network reset each run)")
    ap.add_argument("--stp-U", type=float, default=0.5)
    ap.add_argument("--stp-tau-d-ms", type=float, default=200)
    ap.add_argument("--stp-tau-f-ms", type=float, default=500)
    ap.add_argument("--no-bg", action="store_true", help="Disable background Poisson noise")
    args = ap.parse_args()

    params = deepcopy(default_params)
    params["t_run"] = args.phase_ms * ms  # not used directly
    if args.no_bg:
        params["use_bg"] = False
    params["stp_U"] = args.stp_U
    params["stp_tau_d"] = args.stp_tau_d_ms * ms
    params["stp_tau_f"] = args.stp_tau_f_ms * ms

    stp_pre = set(SUGAR_NEURONS) | set(WATER_NEURONS) | CB0248_IDS | CB0192_IDS
    stp_post = CB0248_IDS | CB0192_IDS | {MN9_ID}

    summary = []
    for i in range(args.n_run):
        exp = f"{args.exp_name}_run{i}"
        out = run_sequence(
            freq1=args.freq1,
            freq2=args.freq2,
            phase_ms=args.phase_ms,
            path_comp=args.path_comp,
            path_con=args.path_con,
            params=params,
            stp_pre=stp_pre,
            stp_post=stp_post,
            exp_name=exp,
            path_res=args.path_res,
            phase1_input=args.phase1_input,
            phase2_input=args.phase2_input,
        )
        total, p1, p2 = summarize(out, args.phase_ms)
        summary.append({"run": i, "exp": exp, "total": total, "phase1": p1, "phase2": p2})
        print(f"[{i}] Saved: {out} | MN9 total {total}, phase1 {p1}, phase2 {p2}")

    # save summary CSV
    Path(args.path_res).mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(Path(args.path_res) / f"{args.exp_name}_summary.csv", index=False)
    print("Summary saved to", Path(args.path_res) / f"{args.exp_name}_summary.csv")


if __name__ == "__main__":
    main()
