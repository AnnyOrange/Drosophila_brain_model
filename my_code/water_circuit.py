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


# Default water GRN list from figures.ipynb (labellar water-sensing neurons, right hemisphere)
WATER_NEURONS: List[int] = [
    720575940612950568,
    720575940631898285,
    720575940606002609,
    720575940612579053,
    720575940622902535,
    720575940616177458,
    720575940660292225,
    720575940622486922,
    720575940613786774,
    720575940629852866,
    720575940625861168,
    720575940613996959,
    720575940617857694,
    720575940644965399,
    720575940625203504,
    720575940630553415,
    720575940635172191,
    720575940634796536,
]


@dataclass
class Paths:
    path_res: Path
    path_comp: Path
    path_con: Path
    n_proc: int = -1


def _mk_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_water_frequency_sweep(
    freqs: Sequence[int],
    paths: Paths,
    water_ids: Sequence[int] = WATER_NEURONS,
    force_overwrite: bool = False,
    base_params: dict | None = None,
) -> None:
    """Run water GRN activation across frequencies."""
    params = deepcopy(base_params or default_params)
    _mk_outdir(paths.path_res)
    for f in freqs:
        params["r_poi"] = f * Hz
        run_exp(
            exp_name=f"waterR_{f}Hz",
            neu_exc=list(water_ids),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Water circuit experiment helper (user code)")
    parser.add_argument(
        "--path-res",
        default="results/my_code_water",
        help="Output directory for parquet/plots",
    )
    parser.add_argument(
        "--path-comp",
        default="2023_03_23_completeness_630_final.csv",
        help="Completeness CSV (switch to 783 by passing the other file)",
    )
    parser.add_argument(
        "--path-con",
        default="2023_03_23_connectivity_630_final.parquet",
        help="Connectivity parquet (switch to 783 by passing the other file)",
    )
    parser.add_argument(
        "--n-proc",
        type=int,
        default=-1,
        help="Number of CPU cores for simulations (-1 uses all available)",
    )
    parser.add_argument(
        "--n-run",
        type=int,
        default=None,
        help="Override number of trials per experiment (default from model.default_params)",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # 创建一个父解析器，存放通用参数
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--no-bg", action="store_true", help="Disable background Poisson noise")

    freq = subparsers.add_parser("freq", parents=[parent_parser], help="Run water activation across frequencies")
    freq.add_argument(
        "--freqs",
        type=int,
        nargs="+",
        default=[20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260],
        help="Frequencies (Hz) to simulate",
    )
    freq.add_argument("--force-overwrite", action="store_true", help="Overwrite outputs")

    plot = subparsers.add_parser("plot", parents=[parent_parser], help="Plot summaries from existing parquet files")
    plot.add_argument(
        "--glob",
        default="waterR*.parquet",
        help="Glob pattern under path_res to load parquet files",
    )
    plot.add_argument(
        "--mn-target",
        type=int,
        default=720575940660219265,
        help="Neuron ID to highlight (e.g., MN9 from the paper)",
    )
    plot.add_argument(
        "--out-prefix",
        default="results/my_code_water/plots/water",
        help="Prefix for saved plots (heatmap + single neuron bar plot)",
    )

    return parser.parse_args()


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

    if args.cmd == "freq":
        run_water_frequency_sweep(
            freqs=args.freqs,
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
            flyid2name={f: f"water_{i+1}" for i, f in enumerate(WATER_NEURONS)},
        )
        out_prefix = Path(args.out_prefix)
        plot_population_heatmap(
            df_rate,
            out_png=out_prefix.with_suffix(".heatmap.png"),
            title="Water GRN activation responses",
        )
        plot_single_neuron_rate(
            df_rate,
            neuron_id=args.mn_target,
            out_png=out_prefix.with_suffix(".mn_target.png"),
            title=f"Neuron {args.mn_target} firing across experiments",
        )
        df_rate.to_csv(out_prefix.with_suffix(".rates.csv"))
        df_rate_std.to_csv(out_prefix.with_suffix(".rate_std.csv"))
    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
