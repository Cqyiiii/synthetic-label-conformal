#!/usr/bin/env python3
"""Regenerate the LaTeX tables used by the V13 manuscript.

The script reads frozen production outputs only. It performs no fitting and no
post-hoc method selection. Method ordering follows the manuscript: the core
RCP-PPI method is shown last within the RCP family.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

import pandas as pd

DATASETS = ["bike", "diamond", "gas_turbine", "naval", "SGEMM", "superconduct", "Transcoding", "WEC"]
DISPLAY = {
    "bike": "Bike",
    "diamond": "Diamond",
    "gas_turbine": "Gas Turbine",
    "naval": "Naval",
    "SGEMM": "SGEMM",
    "superconduct": "Supercond.",
    "Transcoding": "Transcoding",
    "WEC": "WEC",
}
METHODS = [
    "Split", "SPPI", "NNM",
    "CQR", "CQR-PPI",
    "RCP", "RCP-PTFT", "RCP-PPI-CV", "RCP-Aug", "RCP-PPI",
]
SYNTHETIC_METHODS = ["SPPI", "NNM", "CQR-PPI", "RCP-PTFT", "RCP-PPI-CV", "RCP-Aug", "RCP-PPI"]
FAMILY_BREAKS = {"NNM", "CQR-PPI"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "completed_v11_results",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "paper_tables",
    )
    return parser.parse_args()


def parse_cell(cell: str) -> tuple[float, float]:
    left, right = str(cell).split("±")
    return float(left.strip()), float(right.strip())


def matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).set_index("method")
    return df.loc[METHODS, DATASETS]


def fmt_cell(mean: float, sd: float, decimals: int, scale: float = 1.0) -> str:
    return f"{mean * scale:.{decimals}f}$\\pm${sd * scale:.{decimals}f}"


def table_rows(df: pd.DataFrame, *, decimals: int, scale: float, best: str) -> list[str]:
    parsed = {(m, d): parse_cell(df.loc[m, d]) for m in METHODS for d in DATASETS}
    if best == "min":
        best_method = {d: min(METHODS, key=lambda m: parsed[(m, d)][0]) for d in DATASETS}
    elif best == "max":
        best_method = {d: max(METHODS, key=lambda m: parsed[(m, d)][0]) for d in DATASETS}
    else:
        raise ValueError(best)
    rows: list[str] = []
    for method in METHODS:
        vals = []
        for dataset in DATASETS:
            mean, sd = parsed[(method, dataset)]
            cell = fmt_cell(mean, sd, decimals, scale)
            if best_method[dataset] == method:
                cell = rf"\textbf{{{cell}}}"
            vals.append(cell)
        label = rf"\textbf{{{method}}}" if method == "RCP-PPI" else method
        rows.append(label + " & " + " & ".join(vals) + r" \\")
        if method in FAMILY_BREAKS:
            rows.append(r"\midrule")
    return rows


def simple_rows(df: pd.DataFrame, decimals: int) -> str:
    out: list[str] = []
    for method in METHODS:
        vals = []
        for dataset in DATASETS:
            mean, sd = parse_cell(df.loc[method, dataset])
            vals.append(fmt_cell(mean, sd, decimals, 1.0))
        label = rf"\textbf{{{method}}}" if method == "RCP-PPI" else method
        out.append(label + " & " + " & ".join(vals) + r" \\")
        if method in FAMILY_BREAKS:
            out.append(r"\midrule")
    return "\n".join(out)


def header() -> str:
    return "Method & " + " & ".join(DISPLAY[d] for d in DATASETS) + r" \\" 


def standalone_table(*, caption: str, label: str, body: str, best_note: str = "") -> str:
    h = header()
    extra = (" " + best_note.strip()) if best_note.strip() else ""
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{{caption}{extra}}}
\label{{{label}}}
\scriptsize
\setlength{{\tabcolsep}}{{3.1pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lcccccccc}}
\toprule
{h}
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table*}}
""".strip() + "\n"


def build_main(report_dir: Path) -> str:
    msce = matrix(report_dir / "main_absolute_msce_30_raw.csv")
    wsc = matrix(report_dir / "main_absolute_wsc.csv")
    ert = matrix(report_dir / "main_absolute_ert_l1_gain.csv")
    msce_rows = "\n".join(table_rows(msce, decimals=2, scale=1000.0, best="min"))
    wsc_rows = "\n".join(table_rows(wsc, decimals=3, scale=1.0, best="max"))
    ert_rows = "\n".join(table_rows(ert, decimals=3, scale=1.0, best="min"))
    return "\n\n".join([
        standalone_table(
            caption="Grouped mean-squared conditional-coverage error on the eight benchmarks (mean $\\pm$ standard deviation over 30 common splits). Values are multiplied by $10^3$; lower is better.",
            label="tab:main_msce",
            body=msce_rows,
            best_note="Horizontal rules separate method families, and bold marks the best value in each dataset.",
        ).strip(),
        standalone_table(
            caption="Worst-Slice Coverage on the same 30 splits; higher is better.",
            label="tab:main_wsc",
            body=wsc_rows,
            best_note="Horizontal rules separate method families, and bold marks the best value in each dataset.",
        ).strip(),
        standalone_table(
            caption="$L_1$ Excess Risk of Target coverage on the same 30 splits; lower is better.",
            label="tab:main_ert",
            body=ert_rows,
            best_note="Horizontal rules separate method families, and bold marks the best value in each dataset.",
        ).strip(),
    ]) + "\n"


def build_ablation(report_dir: Path) -> str:
    df = pd.read_csv(report_dir / "pairing_ablation_direct.csv")
    rows = []
    for _, row in df.iterrows():
        label = DISPLAY.get(row.dataset, str(row.dataset).title())
        ms30 = f"{row.msce30_diff_mean*1000:.2f} [{row.msce30_diff_low*1000:.2f}, {row.msce30_diff_high*1000:.2f}]"
        ms10 = f"{row.msce10_diff_mean*1000:.2f} [{row.msce10_diff_low*1000:.2f}, {row.msce10_diff_high*1000:.2f}]"
        wsc = f"{row.wsc_diff_mean*100:.2f} [{row.wsc_diff_low*100:.2f}, {row.wsc_diff_high*100:.2f}]"
        rows.append(f"{label} & {ms30} & {ms10} & {wsc} " + "\\\\")
    body = "\n".join(rows)
    return rf"""
\begin{{table}}[t]
\centering
\caption{{Genuine RCP augmentation versus a matched permutation placebo. The placebo preserves the synthetic-score marginal distribution, sample size, architecture, and optimization, but breaks the covariate--score pairing. Entries are paired differences (genuine minus permuted) with 95\% bootstrap intervals over 20 common splits. Negative MSCE and positive WSC favor genuine pairing.}}
\label{{tab:pairing_ablation}}
\scriptsize
\setlength{{\tabcolsep}}{{3.0pt}}
\begin{{tabular}}{{lccc}}
\toprule
Dataset & $\Delta$MSCE$_{{30}}$ ($\times10^3$) & $\Delta$MSCE$_{{10}}$ ($\times10^3$) & $\Delta$WSC ($\times100$) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
""".strip() + "\n"


def build_appendix(report_dir: Path) -> str:
    paired = pd.read_csv(report_dir / "paired_effects.csv")
    paired = paired[paired["suite"] == "v11_main"].copy()
    rows = []
    for method in SYNTHETIC_METHODS:
        cells = []
        for dataset in DATASETS:
            row = paired[(paired.method == method) & (paired.dataset == dataset)].iloc[0]
            x = row.msce_30_raw_diff_mean * 1000
            cell = f"{x:+.2f}"
            if row.msce_30_raw_diff_high < 0:
                cell += r"$^\dagger$"
            cells.append(cell)
        label = rf"\textbf{{{method}}}" if method == "RCP-PPI" else method
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    paired_rows = "\n".join(rows)

    cov = matrix(report_dir / "main_absolute_coverage.csv")
    vol = matrix(report_dir / "main_absolute_mean_log_volume.csv")
    cov_rows = simple_rows(cov, 3)
    vol_rows = simple_rows(vol, 2)
    return "\n\n".join([
        standalone_table(
            caption="Paired change in grouped MSCE relative to the supervised method in the same family (synthetic minus supervised, multiplied by $10^3$). Negative is better. A dagger indicates that the descriptive paired-bootstrap 95\\% interval lies entirely below zero.",
            label="tab:paired_msce_appendix",
            body=paired_rows,
            best_note="Comparisons are dataset specific; effects are not averaged across datasets.",
        ).strip(),
        standalone_table(
            caption="Marginal coverage in the main benchmark (mean $\\pm$ standard deviation over 30 splits). The target is 0.9.",
            label="tab:marginal_coverage_appendix",
            body=cov_rows,
        ).strip(),
        standalone_table(
            caption="Mean log volume of the prediction sets in the main benchmark (mean $\\pm$ standard deviation over 30 splits). Volume is reported on the original outcome scale and is comparable only within a dataset. No method produced a full prediction set.",
            label="tab:log_volume_appendix",
            body=vol_rows,
        ).strip(),
    ]) + "\n"


def build_sensitivity(report_dir: Path) -> str:
    df = pd.read_csv(report_dir / "task_method_rows.csv")
    methods = ["CQR-PPI", "RCP-Aug", "RCP-PPI"]

    teacher_parts = []
    for strength in ["weak", "oracle"]:
        x = df[(df.suite == "v11_teacher_quality") & (df.teacher_strength == strength) & (df.method.isin(methods))]
        teacher_parts.append(x)
    moderate = df[(df.suite == "v11_main") & (df.seed < 10) & (df.method.isin(methods))].copy()
    moderate["teacher_strength"] = "moderate"
    tdf = pd.concat(teacher_parts + [moderate], ignore_index=True)
    tmean = tdf.groupby(["dataset", "method", "teacher_strength"])["msce_30_raw"].mean()
    trows = []
    for dataset in DATASETS:
        vals = ["/".join(f"{tmean.loc[(dataset, method, s)]*1000:.2f}" for s in ["weak", "moderate", "oracle"]) for method in methods]
        trows.append(DISPLAY[dataset] + " & " + " & ".join(vals) + r" \\")
    teacher_rows = "\n".join(trows)

    bdf = df[(df.suite == "v11_label_budget") & (df.method.isin(methods))]
    low = df[(df.suite == "v11_main") & (df.seed < 10) & (df.method.isin(methods))].copy()
    low["label_budget"] = "low"
    bdf = pd.concat([bdf, low], ignore_index=True)
    counterpart = {"CQR-PPI": "CQR", "RCP-PPI": "RCP", "RCP-Aug": "RCP"}
    values = []
    for _, row in bdf.iterrows():
        cp = counterpart[row.method]
        candidates = df[(df.dataset == row.dataset) & (df.seed == row.seed) & (df.method == cp) &
                        (df.teacher_strength == row.teacher_strength) & (df.synthetic_prop == row.synthetic_prop) &
                        (df.base_prop == row.base_prop) & (df.conformal_prop == row.conformal_prop)]
        candidates = candidates[candidates.suite == ("v11_main" if row.label_budget == "low" else "v11_label_budget")]
        if len(candidates) != 1:
            raise RuntimeError(f"Counterpart match failed for {row.method}, {row.dataset}, {row.seed}, {row.label_budget}: {len(candidates)}")
        values.append(row.msce_30_raw - candidates.iloc[0].msce_30_raw)
    bdf = bdf.copy()
    bdf["paired_diff"] = values
    bmean = bdf.groupby(["dataset", "method", "label_budget"])["paired_diff"].mean()
    brows = []
    for dataset in ["bike", "diamond", "SGEMM", "Transcoding"]:
        for method in methods:
            vals = [f"{bmean.loc[(dataset, method, b)]*1000:.2f}" for b in ["ultra_low", "low", "moderate"]]
            label = rf"\textbf{{{method}}}" if method == "RCP-PPI" else method
            brows.append(DISPLAY[dataset] + " & " + label + " & " + " & ".join(vals) + r" \\")
        if dataset != "Transcoding":
            brows.append(r"\midrule")
    budget_rows = "\n".join(brows)

    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Effect of synthetic-label quality. Entries are grouped MSCE $\times10^3$ for weak/moderate/oracle labelers on the same 10 splits; lower is better. Oracle labels use hidden outcomes and are an upper-bound diagnostic.}}
\label{{tab:teacher_sensitivity}}
\scriptsize
\begin{{tabular}}{{lccc}}
\toprule
Dataset & CQR-PPI & RCP-Aug & \textbf{{RCP-PPI}} \\
\midrule
{teacher_rows}
\bottomrule
\end{{tabular}}
\end{{table*}}

\begin{{table*}}[t]
\centering
\caption{{Trusted-label budget sensitivity. Entries are paired grouped-MSCE changes $\times10^3$ relative to the supervised method in the same family; negative is better. Ultra-low, low, and moderate use base/conformal proportions of 1\%/0.5\%, 2\%/1\%, and 4\%/2\%, respectively, on the same 10-seed support.}}
\label{{tab:budget_sensitivity}}
\scriptsize
\begin{{tabular}}{{llrrr}}
\toprule
Dataset & Method & Ultra-low & Low & Moderate \\
\midrule
{budget_rows}
\bottomrule
\end{{tabular}}
\end{{table*}}
""".strip() + "\n"


def build_supplementary(results_root: Path, report_dir: Path) -> str:
    # Intermediate pinball-risk comparison from the trusted CV folds used by RCP-PPI-CV.
    pinball: dict[str, list[tuple[float, float]]] = {d: [] for d in DATASETS}
    for path in glob.glob(str(results_root / "outputs" / "v11_main" / "tasks" / "*.json")):
        result = json.loads(Path(path).read_text())["result"]
        method = next(row for row in result["methods"] if row["method"] == "RCP-PPI-CV")
        losses = method.get("cv_losses") or {}
        if "0.0" in losses and "1.0" in losses:
            pinball[result["dataset"]].append((float(losses["0.0"]), float(losses["1.0"])))
    prows = []
    for dataset in DATASETS:
        arr = pinball[dataset]
        if len(arr) != 30:
            raise RuntimeError(f"Expected 30 CV-loss pairs for {dataset}, found {len(arr)}")
        erm = [x[0] for x in arr]
        ppi = [x[1] for x in arr]
        gain = [x[0] - x[1] for x in arr]
        win = sum(x[1] < x[0] for x in arr) / len(arr)
        prows.append(
            f"{DISPLAY[dataset]} & {statistics.mean(erm):.4f}$\\pm${statistics.stdev(erm):.4f} & "
            f"{statistics.mean(ppi):.4f}$\\pm${statistics.stdev(ppi):.4f} & "
            f"{statistics.mean(gain):.4f}$\\pm${statistics.stdev(gain):.4f} & {win:.2f} " + r"\\"
        )
    pinball_rows = "\n".join(prows)

    # Vary N while keeping every non-pool role fixed. At 0%, use supervised RCP from
    # the 5% pool task for each seed; pool size is irrelevant for the supervised fit.
    df = pd.read_csv(report_dir / "task_method_rows.csv")
    pool = df[df.suite == "v11_pool_size"].copy()
    pool_datasets = ["bike", "diamond", "SGEMM", "Transcoding"]
    nrows = []
    for dataset in pool_datasets:
        base = pool[(pool.dataset == dataset) & (pool.synthetic_prop == 0.05) & (pool.method == "RCP")]
        if base.seed.nunique() != 10:
            raise RuntimeError(f"Missing RCP baseline for {dataset}")
        for share, method in [(0.0, "RCP"), (0.05, "RCP-PPI"), (0.10, "RCP-PPI"), (0.20, "RCP-PPI"), (0.40, "RCP-PPI")]:
            x = base if share == 0.0 else pool[(pool.dataset == dataset) & (pool.synthetic_prop == share) & (pool.method == method)]
            vals = []
            for col, scale, decimals in [("msce_30_raw", 1000.0, 2), ("wsc", 1.0, 3), ("ert_l1_gain", 1.0, 3)]:
                a = pd.to_numeric(x[col]).to_list()
                vals.append(f"{statistics.mean(a)*scale:.{decimals}f}$\\pm${statistics.stdev(a)*scale:.{decimals}f}")
            nrows.append(f"{DISPLAY[dataset]} & {int(round(share*100))}\\% & " + " & ".join(vals) + r" \\")
        if dataset != pool_datasets[-1]:
            nrows.append(r"\midrule")
    n_rows = "\n".join(nrows)

    return rf"""
\begin{{table}}[H]
\centering
\caption{{Intermediate pinball-risk comparison for the RCP score-quantile learner. ERM and unit PPI-ERM are evaluated on the same trusted validation folds used for power selection. We report mean $\pm$ standard deviation over 30 common splits; a positive gain means lower validation risk under PPI-ERM.}}
\label{{tab:pinball_comparison}}
\scriptsize
\begin{{tabular}}{{lcccc}}
\toprule
Dataset & ERM risk & PPI-ERM risk & Risk reduction & PPI-ERM win rate \\
\midrule
{pinball_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Effect of the synthetic-pool size on RCP-PPI. Only the synthetic-pool prefix varies; trusted learning, final calibration, grouping, and evaluation roles are fixed. The 0\% row is supervised RCP. Entries are mean $\pm$ standard deviation over 10 common splits. MSCE is multiplied by $10^3$ (lower is better), WSC is higher-is-better, and $L_1$-ERT is lower-is-better.}}
\label{{tab:varying_N}}
\scriptsize
\begin{{tabular}}{{llccc}}
\toprule
Dataset & Synthetic share & Grouped MSCE & WSC & $L_1$-ERT \\
\midrule
{n_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
""".strip() + "\n"


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    report_dir = (results_root / "reports" / "v11").resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "tables_main.tex": build_main(report_dir),
        "table_ablation.tex": build_ablation(report_dir),
        "tables_appendix.tex": build_appendix(report_dir),
        "tables_sensitivity.tex": build_sensitivity(report_dir),
        "tables_supplementary.tex": build_supplementary(results_root, report_dir),
    }
    supplementary = outputs.pop("tables_supplementary.tex")
    split_marker = "\n\n\\begin{table}[H]\n\\centering\n\\caption{Effect of the synthetic-pool size"
    first, second_tail = supplementary.split(split_marker, 1)
    second = "\\begin{table}[H]\n\\centering\n\\caption{Effect of the synthetic-pool size" + second_tail
    outputs["table_pinball.tex"] = first.strip() + "\n"
    outputs["table_varying_N.tex"] = second.strip() + "\n"
    for name, text in outputs.items():
        (out / name).write_text(text)
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
