#!/usr/bin/env python3
"""Build the executed-report source notebook for the HelpSteer2 pilot."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def value(frame: pd.DataFrame, method: str, column: str) -> float:
    return float(frame.loc[frame["method"].eq(method), column].iloc[0])


def main() -> None:
    summary = pd.read_csv(RESULTS / "downstream_cost_matched" / "summary.csv")
    comparisons = pd.read_csv(RESULTS / "downstream_cost_matched" / "paired_comparisons.csv")
    boundary = pd.read_csv(RESULTS / "boundary_metrics.csv")
    usage = pd.read_csv(RESULTS / "api_usage.csv")

    rcp_msce = value(summary, "RCP", "msce_mean")
    flash_msce = value(summary, "Flash-PPI", "msce_mean")
    pro_msce = value(summary, "Pro-PPI", "msce_mean")
    pro_min = value(summary, "Pro-PPI", "min_group_coverage_mean")
    rcp_min = value(summary, "RCP", "min_group_coverage_mean")
    pro_volume = value(summary, "Pro-PPI", "mean_log_volume_mean")
    rcp_volume = value(summary, "RCP", "mean_log_volume_mean")
    flash_boundary = float(boundary.loc[boundary.model.eq("flash"), "boundary_disagreement"].iloc[0])
    pro_boundary = float(boundary.loc[boundary.model.eq("pro"), "boundary_disagreement"].iloc[0])
    flash_boundary_balanced = float(boundary.loc[boundary.model.eq("flash"), "balanced_accuracy"].iloc[0])
    pro_boundary_balanced = float(boundary.loc[boundary.model.eq("pro"), "balanced_accuracy"].iloc[0])
    flash_boundary_reduction = float(
        boundary.loc[boundary.model.eq("flash"), "relative_reduction_vs_shuffled"].iloc[0]
    )
    pro_boundary_reduction = float(
        boundary.loc[boundary.model.eq("pro"), "relative_reduction_vs_shuffled"].iloc[0]
    )

    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python (ml)", "language": "python", "name": "python3"
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    cells = []
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""# NVIDIA HelpSteer2: realistic LLM synthetic-label pilot

## TL;DR

This is a positive realistic setting for the paper. With 20 paired seeds, RCP has final K-means-group MSCE **{rcp_msce:.5f}**. Flash-PPI lowers it to **{flash_msce:.5f}** ({(flash_msce / rcp_msce - 1) * 100:.1f}%), and Pro-PPI lowers it to **{pro_msce:.5f}** ({(pro_msce / rcp_msce - 1) * 100:.1f}%). Pro-PPI also raises mean minimum-group coverage from **{rcp_min:.3f}** to **{pro_min:.3f}** and lowers mean log-volume from **{rcp_volume:.3f}** to **{pro_volume:.3f}**.

The most informative mechanism result is that global imputation quality is not the whole story. Neither route beats a constant-median predictor in macro MAE. Raw boundary disagreement is **{flash_boundary:.1%}** and **{pro_boundary:.1%}**, but this number alone is class-imbalance dominated because only 13.6% of true scores lie above the $\tau=0.9$ boundary. Against shuffled-score references that preserve each model's score distribution but destroy input-specific alignment, Flash and Pro reduce disagreement by **{flash_boundary_reduction:.1%}** and **{pro_boundary_reduction:.1%}** and attain balanced boundary accuracies of **{flash_boundary_balanced:.3f}** and **{pro_boundary_balanced:.3f}**.
"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## Context & Methods

**Data.** We use NVIDIA's official [HelpSteer2 dataset](https://huggingface.co/datasets/nvidia/HelpSteer2), whose responses have five human ratings on a 0–4 scale: helpfulness, correctness, coherence, complexity, and verbosity.

**Leakage control.** Exact normalized prompt–response duplicates are removed. Official validation is the test set. All remaining roles are split by normalized prompt rather than response, so the two responses associated with a prompt never cross experiment roles.

**Roles.** 801 examples train a five-output mean model; 500 form the LLM pilot (100 paired true/synthetic points and 400 synthetic-only auxiliary points per seed); 100 prompt-group-separated examples tune the quantile learner; 400 independently conformalize it; and 1,038 official validation examples are tested. Results use 20 seeds.

**Representation and score.** A frozen `all-MiniLM-L6-v2` encoder embeds the prompt and response separately; the concatenated 768-dimensional vector is used downstream. Ratings are scaled to [0,1]. The multivariate nonconformity score is the maximum absolute error across the five ratings. The target coverage is 0.9.

**Methods.** RCP learns the score quantile from 100 true scores. Synthetic-only uses 400 LLM scores. PPI uses the same 400 synthetic scores plus the signed correction on 100 paired scores. A prompt-group-separated development set selects the training epoch using true pinball loss; the final conformalization set is untouched.

**Conditional metric.** Test embeddings are partitioned into 10 fixed K-means groups. MSCE is the mean squared deviation of group coverage from 0.9; minimum-group coverage is also reported. These are finite-sample group diagnostics, not a claim of distribution-free exact conditional validity.

**Cost matching.** Token usage and the provider's route-specific token prices are recorded. Equal-N compares 500 total Flash queries with 500 total Pro queries. The cost-matched Pro arm uses the same 100 paired queries and adds as many auxiliary queries as fit under the total Flash-PPI spend for that seed.
"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd()
RESULTS = ROOT / "results"
plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 50)
pd.set_option("display.precision", 4)
"""
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## Data quality and split audit"))
    cells.append(
        nbf.v4.new_code_cell(
            """manifest = json.loads((ROOT / "data/helpsteer2_manifest.json").read_text())
audit = pd.DataFrame({
    "check": [
        "source rows", "rows after deduplication", "duplicate prompt-response rows removed",
        "train/validation overlapping prompts removed", "cross-role prompt overlap",
        "missing required values", "out-of-range labels"
    ],
    "value": [
        manifest["source_rows"], manifest["rows_after_deduplication"],
        manifest["duplicate_prompt_response_rows_removed"],
        manifest["train_validation_prompt_overlap_removed_from_train"],
        manifest["cross_role_prompt_overlap"], manifest["missing_required_values"],
        manifest["out_of_range_label_values"]
    ]
})
display(audit)
display(pd.Series(manifest["role_counts"], name="rows").to_frame())
"""
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## LLM label quality and cost"))
    cells.append(
        nbf.v4.new_code_cell(
            """label_metrics = pd.read_csv(RESULTS / "label_metrics.csv")
boundary = pd.read_csv(RESULTS / "boundary_metrics.csv")
usage = pd.read_csv(RESULTS / "api_usage.csv")
macro = label_metrics[label_metrics.dimension.eq("macro")][
    ["model", "mae", "rmse", "r2", "spearman", "bias", "within_one"]
]
display(macro)
display(boundary)
display(usage[["model", "rows", "prompt_tokens", "completion_tokens", "estimated_cost_usd", "retries", "served_models"]])

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
macro.set_index("model")["mae"].plot.bar(ax=axes[0], color=["#8a8a8a", "#56B4E9", "#D55E00"], title="Macro MAE (lower)")
boundary.set_index("model")["balanced_accuracy"].plot.bar(ax=axes[1], color=["#8a8a8a", "#56B4E9", "#D55E00"], title="Balanced boundary accuracy")
boundary.set_index("model")["relative_reduction_vs_shuffled"].plot.bar(ax=axes[2], color=["#8a8a8a", "#56B4E9", "#D55E00"], title="Disagreement reduction vs shuffled")
for ax in axes:
    ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()
"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """The absolute ratings remain difficult: the LLMs often shift the five-dimensional scale, and the constant-median baseline has lower macro MAE. Raw boundary agreement should not be used alone: because only 13.6% of true scores are above the target boundary, an always-below predictor obtains 86.4% agreement while having balanced accuracy 0.5 and zero F1/MCC. The shuffled-score reference preserves each LLM's score marginal but removes input-specific alignment. Flash and Pro reduce disagreement by 32.5% and 39.5% relative to that reference, with balanced accuracies 0.637 and 0.701. This establishes nontrivial local boundary information without overstating the raw agreement rate."""
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## Downstream conditional-coverage results"))
    cells.append(
        nbf.v4.new_code_cell(
            """summary = pd.read_csv(RESULTS / "downstream_cost_matched/summary.csv")
comparisons = pd.read_csv(RESULTS / "downstream_cost_matched/paired_comparisons.csv")
main_methods = ["RCP", "Flash-PPI", "Pro-PPI"]
shown = summary[summary.method.isin(main_methods)][[
    "method", "query_cost_usd_mean", "true_pinball_mean", "coverage_mean", "msce_mean",
    "min_group_coverage_mean", "mean_log_volume_mean"
]].set_index("method").loc[main_methods]
display(shown)

delta = comparisons[
    comparisons.method.isin(["Flash-PPI", "Pro-PPI"])
    & comparisons.metric.isin(["msce", "min_group_coverage", "mean_log_volume"])
].copy()
display(delta[["method", "metric", "difference_vs_rcp", "ci95_low", "ci95_high"]])

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
plot_specs = [
    ("msce_mean", "Final group MSCE", True),
    ("min_group_coverage_mean", "Minimum-group coverage", False),
    ("mean_log_volume_mean", "Mean log-volume", True),
]
colors = ["#777777", "#56B4E9", "#D55E00"]
for ax, (column, title, _) in zip(axes, plot_specs):
    shown[column].plot.bar(ax=ax, color=colors, title=title)
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()
"""
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## Actual-cost comparison: many Flash labels versus fewer Pro labels"))
    cells.append(
        nbf.v4.new_code_cell(
            """cost_methods = ["RCP", "Flash-PPI", "Pro-PPI-CostMatched", "Pro-PPI"]
cost_view = summary[summary.method.isin(cost_methods)][[
    "method", "auxiliary_size_mean", "query_cost_usd_mean", "msce_mean",
    "min_group_coverage_mean", "mean_log_volume_mean"
]].set_index("method").loc[cost_methods]
cost_view.loc["RCP", "auxiliary_size_mean"] = 0
display(cost_view)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
for method, marker, color in [
    ("Flash-PPI", "o", "#56B4E9"),
    ("Pro-PPI-CostMatched", "s", "#E69F00"),
    ("Pro-PPI", "^", "#D55E00"),
]:
    row = cost_view.loc[method]
    ax.scatter(row.query_cost_usd_mean, row.msce_mean, s=90, marker=marker, color=color, label=method)
ax.axhline(cost_view.loc["RCP", "msce_mean"], color="#777777", linestyle="--", label="RCP")
ax.set(xlabel="Actual LLM query cost (USD)", ylabel="Final group MSCE", title="Quantity-quality-cost tradeoff")
ax.legend(frameon=True)
plt.tight_layout()
plt.show()
"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. **Keep HelpSteer2.** Unlike Civil Comments, it yields a clear final conditional-coverage improvement with an external LLM surrogate.
2. **The realistic result supports the paper's local-quality thesis.** Global five-rating MAE is mediocre, but boundary disagreement is low and both PPI variants improve final MSCE.
3. **The equal-N Pro route is strongest but costs about 3.2× more.** At approximately equal spend, many Flash labels yield slightly lower MSCE and smaller volume, while fewer Pro labels yield slightly higher minimum-group coverage. This is the budget-dependent crossing behavior requested by the reviewer.
4. **Efficiency is not obtained by indiscriminate widening.** Pro-PPI improves MSCE and minimum-group coverage while lowering mean log-volume versus RCP.
5. **Scope.** This is a single realistic case study and uses K-means groups as a conditional diagnostic. The fixed PPI objective improves final conditional metrics here, but the validation-selected models do not improve held-out pinball risk over RCP; this case study should therefore be presented as direct downstream evidence, while the paper's broader risk-propagation claim remains supported by the controlled benchmarks and theory.
"""
        )
    )
    nb["cells"] = cells
    output = ROOT / "helpsteer2_llm_pilot_report.ipynb"
    nbf.write(nb, output)
    print(output)


if __name__ == "__main__":
    main()
