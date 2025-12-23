from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
from brian2 import Hz

# ensure repo root on sys.path so we can import model/utils without installing
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import default_params, run_exp  # noqa: E402
import utils as utl  # noqa: E402


# Default sugar GRN list from example notebook (labellar sugar-sensing neurons, right hemisphere)
SUGAR_NEURONS: List[int] = [
    720575940624963786,
    720575940630233916,
    720575940637568838,
    720575940638202345,
    720575940617000768,
    720575940630797113,
    720575940632889389,
    720575940621754367,
    720575940621502051,
    720575940640649691,
    720575940639332736,
    720575940616885538,
    720575940639198653,
    720575940620900446,
    720575940617937543,
    720575940632425919,
    720575940633143833,
    720575940612670570,
    720575940628853239,
    720575940629176663,
    720575940611875570,
]


@dataclass
class Paths:
    path_res: Path
    path_comp: Path
    path_con: Path
    n_proc: int = -1


def _mk_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_sugar_frequency_sweep(
    freqs: Sequence[int],
    paths: Paths,
    sugar_ids: Sequence[int] = SUGAR_NEURONS,
    force_overwrite: bool = False,
    base_params: dict | None = None,
) -> None:
    """Run sugar GRN activation across frequencies."""
    params = deepcopy(base_params or default_params)
    _mk_outdir(paths.path_res)
    for f in freqs:
        params["r_poi"] = f * Hz
        run_exp(
            exp_name=f"sugarR_{f}Hz",
            neu_exc=list(sugar_ids),
            params=params,
            path_res=str(paths.path_res),
            path_comp=str(paths.path_comp),
            path_con=str(paths.path_con),
            n_proc=paths.n_proc,
            force_overwrite=force_overwrite,
        )


def run_silencing_batch(
    silence_ids: Sequence[int],
    freq: int,
    paths: Paths,
    sugar_ids: Sequence[int] = SUGAR_NEURONS,
    force_overwrite: bool = False,
    base_params: dict | None = None,
) -> None:
    """Run sugar activation while silencing one neuron at a time."""
    params = deepcopy(base_params or default_params)
    params["r_poi"] = freq * Hz
    _mk_outdir(paths.path_res)
    for sid in silence_ids:
        run_exp(
            exp_name=f"sugarR-{sid}",
            neu_exc=list(sugar_ids),
            neu_slnc=[sid],
            params=params,
            path_res=str(paths.path_res),
            path_comp=str(paths.path_comp),
            path_con=str(paths.path_con),
            n_proc=paths.n_proc,
            force_overwrite=force_overwrite,
        )


def collect_rates(
    parquet_files: Iterable[Path],
    t_run,
    n_run: int,
    flyid2name=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load parquet files and compute firing rates/std."""
    paths = [str(p) for p in parquet_files]
    df_spike = utl.load_exps(paths)
    return utl.get_rate(df_spike, t_run=t_run, n_run=n_run, flyid2name=flyid2name or {})


def plot_population_heatmap(df_rate: pd.DataFrame, out_png: Path, title: str = "") -> None:
    """Plot a heatmap of firing rates (neurons x experiments)."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise SystemExit("matplotlib and seaborn are required for plotting") from exc

    _mk_outdir(out_png.parent)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_rate.fillna(0), cmap="magma", cbar_kws={"label": "Hz"})
    plt.xlabel("Experiment")
    plt.ylabel("Flywire ID")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_single_neuron_rate(
    df_rate: pd.DataFrame, neuron_id: int, out_png: Path, title: str = ""
) -> None:
    """Plot firing rate of one neuron across experiments."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting") from exc

    _mk_outdir(out_png.parent)
    row = df_rate.loc[neuron_id].fillna(0)
    plt.figure(figsize=(8, 4))
    row.plot(kind="bar")
    plt.ylabel("Rate (Hz)")
    plt.xlabel("Experiment")
    plt.xticks(rotation=45, ha="right")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main() -> None:
    # 采用更鲁棒的方式：定义一个顶层解析器
    parser = argparse.ArgumentParser(description="Sugar circuit experiment helper")
    
    # 将全局可选参数定义在主解析器上
    parser.add_argument("--path-res", default="results/my_code_sugar")
    parser.add_argument("--path-comp", default="2023_03_23_completeness_630_final.csv")
    parser.add_argument("--path-con", default="2023_03_23_connectivity_630_final.parquet")
    parser.add_argument("--n-proc", type=int, default=-1)
    parser.add_argument("--n-run", type=int, default=None)
    parser.add_argument("--no-bg", action="store_true", help="Disable background Poisson noise")

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # freq 子命令
    freq_p = subparsers.add_parser("freq", help="Run frequency sweep")
    freq_p.add_argument("--freqs", type=int, nargs="+", default=[25, 50, 75, 100, 125, 150, 175, 200])
    freq_p.add_argument("--force-overwrite", action="store_true")

    # silence 子命令
    silence_p = subparsers.add_parser("silence", help="Run silencing sweep")
    silence_p.add_argument("--freq", type=int, default=100)
    silence_p.add_argument("--silence-ids", type=int, nargs="+", required=True)
    silence_p.add_argument("--force-overwrite", action="store_true")

    # plot 子命令
    plot_p = subparsers.add_parser("plot", help="Plot results")
    plot_p.add_argument("--glob", default="sugarR*.parquet")
    plot_p.add_argument("--mn-target", type=int, default=720575940660219265)
    plot_p.add_argument("--out-prefix", default="results/my_code_sugar/plots/sugar")

    # 关键修改：支持在子命令之后解析全局参数
    # 如果用户把 --no-bg 放在最后，argparse 默认会报错。
    # 我们在这里手动调整 sys.argv 或者使用 parse_known_args (稍微复杂)
    # 最简单且符合规范的方法：告诉用户全局参数放在中间，或者我们手动移动它。
    
    # 尝试移动 --no-bg 到子命令之前
    argv = sys.argv[1:]
    if "--no-bg" in argv:
        argv.remove("--no-bg")
        argv.insert(0, "--no-bg")
    
    args = parser.parse_args(argv)

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

    if args.cmd == "freq":
        run_sugar_frequency_sweep(
            freqs=args.freqs,
            paths=paths,
            base_params=base_params,
            force_overwrite=args.force_overwrite,
        )
    elif args.cmd == "silence":
        run_silencing_batch(
            silence_ids=args.silence_ids,
            freq=args.freq,
            paths=paths,
            base_params=base_params,
            force_overwrite=args.force_overwrite,
        )
    elif args.cmd == "plot":
        parquet_files = sorted(paths.path_res.glob(args.glob))
        if not parquet_files:
            raise SystemExit(f"No parquet files found in {paths.path_res} matching {args.glob}")
        df_rate, df_rate_std = collect_rates(
            parquet_files=parquet_files,
            t_run=base_params["t_run"],
            n_run=base_params["n_run"],
            flyid2name={f: f"sugar_{i+1}" for i, f in enumerate(SUGAR_NEURONS)},
        )
        out_prefix = Path(args.out_prefix)
        plot_population_heatmap(
            df_rate,
            out_png=out_prefix.with_suffix(".heatmap.png"),
            title="Sugar GRN activation responses",
        )
        plot_single_neuron_rate(
            df_rate,
            neuron_id=args.mn_target,
            out_png=out_prefix.with_suffix(".mn_target.png"),
            title=f"Neuron {args.mn_target} firing across experiments",
        )
        df_rate.to_csv(out_prefix.with_suffix(".rates.csv"))
        df_rate_std.to_csv(out_prefix.with_suffix(".rate_std.csv"))


if __name__ == "__main__":
    main()
