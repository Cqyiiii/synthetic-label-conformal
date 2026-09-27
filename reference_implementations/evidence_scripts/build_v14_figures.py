#!/usr/bin/env python3
"""Build the two V14 paper figures and HelpSteer2 appendix table.

All values are frozen from the completed V11 benchmark / V14 HelpSteer2
reports. The script performs no fitting or post-hoc method selection.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
})
sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)

PAIRING = pd.DataFrame([
    ("Bike", -1.74, -2.75, -0.88, -1.57, -2.65, -0.72, 1.80, 0.20, 3.40),
    ("Diamond", -4.73, -5.78, -3.77, -4.08, -5.04, -3.14, 10.41, 7.89, 12.78),
    ("Naval", -1.13, -2.07, -0.34, -1.06, -1.86, -0.37, 1.93, 0.11, 3.92),
    ("SGEMM", -1.58, -2.11, -0.94, -1.07, -1.53, -0.55, 4.08, 2.91, 5.33),
    ("Transcoding", -1.14, -1.56, -0.73, -0.65, -0.93, -0.40, 2.91, 1.76, 4.28),
], columns=["dataset", "msce30", "msce30_lo", "msce30_hi", "msce10", "msce10_lo", "msce10_hi", "wsc", "wsc_lo", "wsc_hi"])

HELPSTEER = pd.DataFrame([
    ("RCP", 0.0, 0.006652, 0.7382, -0.4264, "Supervised"),
    ("Flash-SynOnly", 0.042180, 0.006476, 0.7416, -0.3696, "Synthetic only"),
    ("Flash-PPI", 0.052841, 0.005016, 0.7631, -0.4342, "PPI"),
    ("Pro-SynOnly", 0.135181, 0.004789, 0.7677, -0.3888, "Synthetic only"),
    ("Pro-PPI", 0.169318, 0.003998, 0.7725, -0.4808, "PPI"),
    ("Pro-PPI\n(cost-matched)", 0.052689, 0.005331, 0.7713, -0.3764, "PPI"),
], columns=["method", "cost", "msce", "min_group", "log_volume", "type"])

HELPSTEER_PAIRED = pd.DataFrame([
    ("Flash-PPI", -0.001636, -0.002944, -0.000333, 0.024853, -0.000612, 0.049369, -0.007762, -0.037472, 0.024171),
    ("Pro-PPI", -0.002654, -0.003468, -0.001789, 0.034287, 0.012665, 0.055582, -0.054313, -0.090882, -0.016796),
    ("Pro-PPI (cost-matched)", -0.001322, -0.002632, -0.000053, 0.033104, 0.016298, 0.049611, 0.050064, 0.009115, 0.089926),
], columns=["method", "msce_diff", "msce_low", "msce_high", "min_group_diff", "min_group_low", "min_group_high", "log_volume_diff", "log_volume_low", "log_volume_high"])

HELPSTEER_BOUNDARY = pd.DataFrame([
    ("Constant median", 0.680, 0.136, 0.136, 0.0, 0.500),
    ("Flash", 0.706, 0.134, 0.198, 32.5, 0.637),
    ("Pro", 0.846, 0.152, 0.251, 39.5, 0.701),
], columns=["labeler", "macro_mae", "observed_disagreement", "shuffled_disagreement", "reduction_percent", "balanced_accuracy"])


def pairing_figure(outdir: Path) -> None:
    palette = sns.color_palette("deep")
    y = list(range(len(PAIRING)))
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.25), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    offset = 0.12
    for vals, lo, hi, marker, label, color, dy in [
        (PAIRING.msce30, PAIRING.msce30_lo, PAIRING.msce30_hi, "o", r"Grouped MSCE, $K=30$", palette[0], -offset),
        (PAIRING.msce10, PAIRING.msce10_lo, PAIRING.msce10_hi, "s", r"Split-half MSCE, $K=10$", palette[1], offset),
    ]:
        yy = [v + dy for v in y]
        ax.errorbar(vals, yy, xerr=[vals - lo, hi - vals], fmt=marker, ms=4.2, lw=1.0, capsize=2.0, label=label, color=color)
    ax.axvline(0, color="0.25", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(PAIRING.dataset)
    ax.invert_yaxis()
    ax.set_xlabel(r"Paired difference in MSCE ($\times 10^3$)")
    ax.legend(loc="lower left", frameon=True)

    ax = axes[1]
    vals, lo, hi = PAIRING.wsc, PAIRING.wsc_lo, PAIRING.wsc_hi
    ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="o", ms=4.5, lw=1.0, capsize=2.0, color=palette[2])
    ax.axvline(0, color="0.25", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel(r"Paired difference in WSC ($\times 100$)")
    sns.despine(fig=fig, left=False, bottom=False)
    fig.subplots_adjust(left=0.115, right=0.995, bottom=0.22, top=0.98, wspace=0.16)
    fig.savefig(outdir / "pairing_ablation.pdf", bbox_inches="tight")
    fig.savefig(outdir / "pairing_ablation.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def helpsteer_figure(outdir: Path) -> None:
    palette = {"Supervised": "#6B7280", "Synthetic only": "#9CA3AF", "PPI": "#3B82F6"}
    y = list(range(len(HELPSTEER)))
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.75), sharey=True)
    specs = [
        ("msce", "Group MSCE", "lower is better"),
        ("min_group", "Minimum-group coverage", "higher is better"),
        ("log_volume", "Mean log-volume", "lower is better"),
    ]
    for ax, (col, title, subtitle) in zip(axes, specs):
        for i, row in HELPSTEER.iterrows():
            ax.scatter(row[col], i, s=34, color=palette[row["type"]], edgecolor="white", linewidth=0.55, zorder=3)
        ax.axvline(HELPSTEER.loc[HELPSTEER.method == "RCP", col].iloc[0], color="0.35", lw=0.8, ls="--", zorder=1)
        ax.set_title(f"{title}\n{subtitle}")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(HELPSTEER.method)
    axes[0].invert_yaxis()
    for ax in axes[1:]:
        ax.set_yticks(y)
    handles = [mpl.lines.Line2D([], [], marker='o', linestyle='', color=palette[k], label=k, markersize=5) for k in ["Supervised", "Synthetic only", "PPI"]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.50, 0.005))
    sns.despine(fig=fig, left=False, bottom=False)
    fig.subplots_adjust(left=0.245, right=0.995, bottom=0.20, top=0.82, wspace=0.18)
    fig.savefig(outdir / "helpsteer2_downstream.pdf", bbox_inches="tight")
    fig.savefig(outdir / "helpsteer2_downstream.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def helpsteer_table(path: Path) -> None:
    text = r"""\begin{table}[H]
\centering
\caption{HelpSteer2 downstream results, averaged over 20 paired seeds. Equal-count arms use 100 paired points and 400 auxiliary labels; the cost-matched Pro arm uses 54.8 auxiliary labels on average. Lower MSCE, true pinball risk, and log-volume are better; higher minimum-group coverage is better.}
\label{tab:helpsteer_downstream}
\small
\setlength{\tabcolsep}{4.4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrr}
\toprule
Method & Aux. $N$ & Query cost & True pinball & Coverage & MSCE & Min-group & Log-volume \\
\midrule
RCP & 0.0 & \$0.0000 & 0.0558 & 0.8794 & 0.00665 & 0.738 & $-0.426$ \\
Flash-SynOnly & 400.0 & \$0.0422 & 0.0520 & 0.8865 & 0.00648 & 0.742 & $-0.370$ \\
Flash-PPI & 400.0 & \$0.0528 & 0.0611 & 0.8899 & 0.00502 & 0.763 & $-0.434$ \\
Pro-SynOnly & 400.0 & \$0.1352 & 0.0507 & 0.8886 & 0.00479 & 0.768 & $-0.389$ \\
Pro-PPI & 400.0 & \$0.1693 & 0.0619 & 0.8937 & 0.00400 & 0.773 & $-0.481$ \\
Pro-PPI (cost-matched) & 54.8 & \$0.0527 & 0.0721 & 0.8846 & 0.00533 & 0.771 & $-0.376$ \\
\bottomrule
\end{tabular}}
\end{table}

\begin{table}[H]
\centering
\caption{Paired HelpSteer2 effects relative to RCP. Entries are mean differences with paired-bootstrap 95\% intervals. Negative MSCE and log-volume and positive minimum-group coverage favor PPI.}
\label{tab:helpsteer_paired}
\small
\setlength{\tabcolsep}{5.0pt}
\resizebox{0.96\textwidth}{!}{%
\begin{tabular}{lrrr}
\toprule
Method & $\Delta$MSCE & $\Delta$ minimum-group coverage & $\Delta$ log-volume \\
\midrule
Flash-PPI & $-0.00164$ [$-0.00294,-0.00033$] & $+0.0249$ [$-0.0006,+0.0494$] & $-0.0078$ [$-0.0375,+0.0242$] \\
Pro-PPI & $-0.00265$ [$-0.00347,-0.00179$] & $+0.0343$ [$+0.0127,+0.0556$] & $-0.0543$ [$-0.0909,-0.0168$] \\
Pro-PPI (cost-matched) & $-0.00132$ [$-0.00263,-0.00005$] & $+0.0331$ [$+0.0163,+0.0496$] & $+0.0501$ [$+0.0091,+0.0899$] \\
\bottomrule
\end{tabular}}
\end{table}

\begin{table}[H]
\centering
\caption{Global rating accuracy and local boundary information on the 500-row HelpSteer2 pilot. Shuffling preserves each synthetic-score marginal while destroying input-specific alignment.}
\label{tab:helpsteer_boundary}
\small
\setlength{\tabcolsep}{5.0pt}
\resizebox{0.88\textwidth}{!}{%
\begin{tabular}{lrrrrr}
\toprule
Labeler & Macro MAE & Observed disagree. & Shuffled disagree. & Reduction vs. shuffled & Balanced acc. \\
\midrule
Constant median & 0.680 & 0.136 & 0.136 & 0.0\% & 0.500 \\
Flash & 0.706 & 0.134 & 0.198 & 32.5\% & 0.637 \\
Pro & 0.846 & 0.152 & 0.251 & 39.5\% & 0.701 \\
\bottomrule
\end{tabular}}
\end{table}
"""
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--table-out", type=Path, default=None)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    PAIRING.to_csv(args.outdir / "pairing_ablation_data.csv", index=False)
    HELPSTEER.to_csv(args.outdir / "helpsteer2_downstream_data.csv", index=False)
    HELPSTEER_PAIRED.to_csv(args.outdir / "helpsteer2_paired_effects.csv", index=False)
    HELPSTEER_BOUNDARY.to_csv(args.outdir / "helpsteer2_boundary_data.csv", index=False)
    pairing_figure(args.outdir)
    helpsteer_figure(args.outdir)
    helpsteer_table(args.table_out or (args.outdir / "table_helpsteer.tex"))


if __name__ == "__main__":
    main()
