from __future__ import annotations

import pandas as pd
from brian2 import Synapses, Network, ms, second

from model import default_params, create_model


def create_model_stp(path_comp: str, path_con: str, params: dict, stp_pre_ids: set[int], stp_post_ids: set[int]):
    """Build network with STP on selected synapses; others remain static."""
    # base neurons/monitor
    neu, _, spk_mon, bg_noise = create_model(path_comp, path_con, params)

    # load connectivity to split
    df_con = pd.read_parquet(path_con)
    df_comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}

    # mask for stp edges
    mask = df_con["Presynaptic_ID"].isin(stp_pre_ids) & df_con["Postsynaptic_ID"].isin(stp_post_ids)
    df_stp = df_con[mask]
    df_base = df_con[~mask]

    # base synapses
    from brian2 import Synapses
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
        name="stp_synapses",
        namespace={
            "U": params.get("stp_U", 0.5),
            "tau_d": params.get("stp_tau_d", 200 * ms),
            "tau_f": params.get("stp_tau_f", 500 * ms),
        },
    )
    syn_stp.connect(i=df_stp["Presynaptic_Index"].values, j=df_stp["Postsynaptic_Index"].values)
    if len(df_stp):
        syn_stp.w_base = df_stp["Excitatory x Connectivity"].values * params["w_syn"]
        syn_stp.u = params.get("stp_u0", params.get("stp_U", 0.5))
        syn_stp.x = 1

    objs = [neu, syn_base, syn_stp, spk_mon]
    if bg_noise is not None:
        objs.append(bg_noise)
    net = Network(*objs)
    return net, neu, syn_base, syn_stp, spk_mon, flyid2i
