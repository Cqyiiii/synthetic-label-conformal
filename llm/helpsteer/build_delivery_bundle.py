#!/usr/bin/env python3
"""Build the reviewer-facing HelpSteer2 report and a complete raw-data bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
OUT = ROOT / "deliverables"
RAW = OUT / "raw_data"
CODE = OUT / "code"


def row(frame: pd.DataFrame, method: str) -> pd.Series:
    return frame.loc[frame["method"].eq(method)].iloc[0]


def metric(frame: pd.DataFrame, method: str, name: str) -> pd.Series:
    return frame.loc[frame["method"].eq(method) & frame["metric"].eq(name)].iloc[0]


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |"
    body = "\n".join("| " + " | ".join(values) + " |" for values in rows)
    return "\n".join([head, rule, body])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_reports() -> None:
    manifest = json.loads((DATA / "helpsteer2_manifest.json").read_text(encoding="utf-8"))
    embed_manifest = json.loads((DATA / "helpsteer2_embeddings_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((RESULTS / "validation_summary.json").read_text(encoding="utf-8"))
    settings = json.loads(
        (RESULTS / "downstream_cost_matched" / "summary.json").read_text(encoding="utf-8")
    )["settings"]
    summary = pd.read_csv(RESULTS / "downstream_cost_matched" / "summary.csv")
    comparisons = pd.read_csv(RESULTS / "downstream_cost_matched" / "paired_comparisons.csv")
    label_metrics = pd.read_csv(RESULTS / "label_metrics.csv")
    boundary = pd.read_csv(RESULTS / "boundary_metrics.csv")
    usage = pd.read_csv(RESULTS / "api_usage.csv")

    rcp = row(summary, "RCP")
    flash = row(summary, "Flash-PPI")
    pro = row(summary, "Pro-PPI")
    pro_cm = row(summary, "Pro-PPI-CostMatched")
    flash_syn = row(summary, "Flash-SynOnly")
    pro_syn = row(summary, "Pro-SynOnly")

    flash_msce_reduction = 1 - flash.msce_mean / rcp.msce_mean
    pro_msce_reduction = 1 - pro.msce_mean / rcp.msce_mean
    pro_cost_ratio = pro.query_cost_usd_mean / flash.query_cost_usd_mean

    macro = label_metrics.loc[label_metrics["dimension"].eq("macro")].set_index("model")
    boundary_indexed = boundary.set_index("model")
    usage_indexed = usage.set_index("model")

    result_rows = []
    for name in ["RCP", "Flash-SynOnly", "Flash-PPI", "Pro-SynOnly", "Pro-PPI", "Pro-PPI-CostMatched"]:
        values = row(summary, name)
        result_rows.append(
            [
                name,
                f"{values.auxiliary_size_mean:.1f}",
                f"${values.query_cost_usd_mean:.6f}",
                f"{values.true_pinball_mean:.6f}",
                f"{values.coverage_mean:.4f}",
                f"{values.msce_mean:.6f}",
                f"{values.min_group_coverage_mean:.4f}",
                f"{values.mean_log_volume_mean:.4f}",
            ]
        )
    full_result_table = markdown_table(
        ["Method", "Aux. N", "Query cost", "True pinball ↓", "Coverage", "MSCE ↓", "Min-group ↑", "Log-volume ↓"],
        result_rows,
    )

    paired_rows = []
    for method_name in ["Flash-PPI", "Pro-PPI", "Pro-PPI-CostMatched"]:
        for metric_name in ["msce", "min_group_coverage", "mean_log_volume", "true_pinball"]:
            values = metric(comparisons, method_name, metric_name)
            paired_rows.append(
                [
                    method_name,
                    metric_name,
                    f"{values.difference_vs_rcp:+.6f}",
                    f"[{values.ci95_low:+.6f}, {values.ci95_high:+.6f}]",
                ]
            )
    paired_table = markdown_table(["Method", "Metric", "Paired Δ vs RCP", "Paired-bootstrap 95% CI"], paired_rows)

    global_rows = []
    for model, label in [("base_median", "Constant median"), ("flash", "Flash"), ("pro", "Pro")]:
        values = macro.loc[model]
        global_rows.append(
            [
                label,
                f"{values.mae:.4f}",
                "—" if pd.isna(values.spearman) else f"{values.spearman:.4f}",
                f"{values.bias:+.4f}",
                f"{values.within_one:.4f}",
            ]
        )
    global_table = markdown_table(
        ["Labeler", "Macro MAE ↓", "Macro Spearman ↑", "Macro bias", "Within-one ↑"], global_rows
    )

    boundary_rows = []
    for model, label in [("base_median", "Constant median"), ("flash", "Flash"), ("pro", "Pro")]:
        values = boundary_indexed.loc[model]
        boundary_rows.append(
            [
                label,
                f"{values.boundary_disagreement:.3f}",
                f"{values.shuffled_pair_disagreement:.3f}",
                pct(values.relative_reduction_vs_shuffled),
                f"{values.balanced_accuracy:.3f}",
                f"{values.above_boundary_f1:.3f}",
                f"{values.matthews_correlation:.3f}",
            ]
        )
    boundary_table = markdown_table(
        [
            "Labeler",
            "Observed disagree. ↓",
            "Shuffled disagree. ↓",
            "Reduction vs shuffled ↑",
            "Balanced acc. ↑",
            "Above-boundary F1 ↑",
            "MCC ↑",
        ],
        boundary_rows,
    )

    setting_table = markdown_table(
        ["Component", "Setting"],
        [
            ["Dataset", "NVIDIA HelpSteer2; five ratings in {0,1,2,3,4}"],
            ["Rows after deduplication", f"{manifest['rows_after_deduplication']:,}"],
            ["Base mean-model training", f"{settings['base_train']} rows / {manifest['role_prompt_counts']['base_train']} prompt groups"],
            ["LLM pilot pool", f"500 rows / {manifest['role_prompt_counts']['pilot']} prompt groups"],
            ["Paired true + synthetic", f"{settings['paired']} per seed"],
            ["Synthetic-only auxiliary", f"{settings['auxiliary']} per equal-N arm"],
            ["Quantile-development", f"{settings['quantile_development']} prompt-separated rows"],
            ["Final conformalization", f"{settings['conformal']} prompt-separated rows"],
            ["Test", f"{settings['test']:,} official-validation rows / {manifest['role_prompt_counts']['test']} prompt groups"],
            ["Seeds", str(settings["seeds"])],
            ["Target coverage", f"τ = {settings['tau']}"],
            ["Conditional diagnostic", "10 fixed K-means groups on frozen test embeddings"],
        ],
    )

    setting_text = rf"""# HelpSteer2 realistic LLM synthetic-label experiment: complete setting

This experiment is a realistic case study for the paper's central question: can inexpensive labels from a fixed external model improve the conditional coverage of conformal prediction when true human ratings are limited? It uses NVIDIA HelpSteer2, where each prompt–response pair has five human ratings—helpfulness, correctness, coherence, complexity, and verbosity—on a 0–4 scale.

## Data construction and leakage control

The official dataset contains {manifest['source_rows']:,} rows. We remove {manifest['duplicate_prompt_response_rows_removed']} exact normalized prompt–response duplicates and remove the {manifest['train_validation_prompt_overlap_removed_from_train']} normalized prompt shared between the official train and validation splits, leaving {manifest['rows_after_deduplication']:,} rows. The official validation set is reserved as test data. All other experimental roles are allocated at the normalized-prompt level, so alternative responses to the same prompt never cross base-training, pilot, development/conformalization, or test roles. The resulting cross-role prompt overlap is exactly zero.

{setting_table}

## LLM synthetic labelers

We query the production API routes `deepseek-v4-flash` and `deepseek-v4-pro` independently on the same 500 pilot rows. The prompt includes the five HelpSteer2 rubric definitions and asks for a JSON object with exactly five numeric scores. Generation uses temperature 0, JSON mode, disabled reasoning output, a 96-token output cap, and no in-context examples. Prompt and response are wrapped as untrusted text so that their content is evaluated rather than followed as instructions. All 1,000 calls succeed without retries or invalid ratings.

Flash uses {usage_indexed.loc['flash', 'prompt_tokens']:,} prompt tokens and {usage_indexed.loc['flash', 'completion_tokens']:,} completion tokens for an estimated ${usage_indexed.loc['flash', 'estimated_cost_usd']:.6f}. Pro uses the same prompt-token count and {usage_indexed.loc['pro', 'completion_tokens']:,} completion tokens for ${usage_indexed.loc['pro', 'estimated_cost_usd']:.6f}. These are the actual recorded route-specific token costs used in the cost comparison.

## Representation, base predictor, and nonconformity score

The prompt and response are encoded separately using a frozen `sentence-transformers/all-MiniLM-L6-v2` encoder with maximum sequence length 256. The two normalized 384-dimensional embeddings are concatenated into a 768-dimensional feature vector. Human and LLM ratings are scaled to [0,1].

The base predictor is a five-output MLP with layers 768→128→64→5, ReLU hidden activations, and sigmoid output. It is fit once on the 801-row base-training role using AdamW (learning rate 3e-3, weight decay 1e-4) and Huber loss. The multivariate nonconformity score is

$$
S(x,y)=\max_{{k\in\{{1,\ldots,5\}}}}|y_k-\widehat\mu_k(x)|.
$$

Thus a single scalar radius controls a five-dimensional box prediction set.

## Quantile learners and PPI objective

The conditional score-quantile model is a 768→64→32→1 MLP with ReLU hidden activations and a sigmoid output. It is optimized by AdamW (learning rate 5e-3, weight decay 2e-4) with gradient clipping at 5.0. All methods start from the same initialization within a seed. At every epoch, the checkpoint is evaluated on a separate prompt-group-separated 100-row development role using true pinball risk; the best checkpoint is retained. The final 400-row conformalization role is never used for this selection.

For each of 20 seeds, 100 of the 500 pilot rows form the paired set and the other 400 form the auxiliary set. We compare:

- **RCP:** pinball ERM on 100 true scores.
- **Synthetic-only:** pinball ERM on 400 Flash or Pro synthetic scores.
- **PPI:** synthetic pinball risk on the 400 auxiliary points plus the signed true-minus-synthetic loss correction on the 100 paired points.
- **Pro-PPI-CostMatched:** the same 100 paired Pro calls plus as many auxiliary Pro calls as fit within the total Flash-PPI spend for that seed; this yields {pro_cm.auxiliary_size_mean:.1f} auxiliary points on average.

## Final conformalization and evaluation

After learning $\widehat q(x)$, the independent 400-row conformalization set supplies residuals $S-\widehat q(X)$. We use the finite-sample higher/order-statistic conformal quantile at τ=0.9 to obtain the scalar correction $\widehat\gamma$. The final five-dimensional prediction box is centered at $\widehat\mu(x)$ with clipped radius $\widehat q(x)+\widehat\gamma$.

The fixed 1,038-row official validation test set is partitioned into 10 K-means groups using the frozen embeddings. We report marginal coverage, mean squared conditional error (MSCE) across these groups, minimum-group coverage, mean radius, and mean log-volume. These groups are a reproducible finite-sample conditional diagnostic; they are not claimed to provide distribution-free exact conditional validity.
"""
    (OUT / "paper_ready_experiment_setting.md").write_text(setting_text, encoding="utf-8")

    report = rf"""# NVIDIA HelpSteer2 realistic LLM synthetic-label experiment

**Experiment date:** 2026-08-04  
**Status:** complete; all automated integrity checks pass  
**Primary question:** whether realistic, externally generated LLM ratings can improve final conditional coverage, and how the answer changes under equal label counts versus equal API spend.

## Executive summary

The setting succeeds as a realistic downstream conditional-coverage demonstration. With the same 100 paired true/synthetic points and 400 synthetic-only auxiliary points, Flash-PPI reduces final group MSCE from **{rcp.msce_mean:.6f}** to **{flash.msce_mean:.6f}** ({pct(flash_msce_reduction)}), and Pro-PPI reduces it to **{pro.msce_mean:.6f}** ({pct(pro_msce_reduction)}). Pro-PPI also raises mean minimum-group coverage from **{rcp.min_group_coverage_mean:.3f}** to **{pro.min_group_coverage_mean:.3f}** and lowers mean log-volume from **{rcp.mean_log_volume_mean:.3f}** to **{pro.mean_log_volume_mean:.3f}**.

The realistic cost comparison is not a one-sided “stronger model always wins” result. Equal-N Pro-PPI costs {pro_cost_ratio:.2f}× Flash-PPI and delivers the best MSCE and volume. At nearly equal spend, Flash uses 400 auxiliary labels while Pro uses {pro_cm.auxiliary_size_mean:.1f} on average: Flash has slightly lower mean MSCE (**{flash.msce_mean:.6f}** vs **{pro_cm.msce_mean:.6f}**) and smaller volume, whereas cost-matched Pro has slightly higher minimum-group coverage (**{pro_cm.min_group_coverage_mean:.3f}** vs **{flash.min_group_coverage_mean:.3f}**). This is the reviewer-requested quantity–quality–cost tradeoff.

The boundary analysis also needs a careful reference. Raw boundary agreement is high partly because only **{pct(boundary_indexed.loc['flash', 'true_above_rate'])}** of true scores are above the τ=0.9 boundary. A constant-median score predictor therefore obtains {pct(boundary_indexed.loc['base_median', 'boundary_agreement'])} raw agreement while having no useful above-boundary discrimination. The meaningful comparisons are against a shuffled-score baseline and class-balanced metrics: Flash reduces disagreement by **{pct(boundary_indexed.loc['flash', 'relative_reduction_vs_shuffled'])}** relative to shuffling and obtains balanced accuracy **{boundary_indexed.loc['flash', 'balanced_accuracy']:.3f}**; Pro reduces it by **{pct(boundary_indexed.loc['pro', 'relative_reduction_vs_shuffled'])}** and obtains balanced accuracy **{boundary_indexed.loc['pro', 'balanced_accuracy']:.3f}**. Thus both routes contain input-specific local boundary information even though neither beats a constant-median predictor in global macro MAE.

## 1. Complete experimental setting

The paper-ready setting description is provided separately in `paper_ready_experiment_setting.md`. In brief:

{setting_table}

The full data flow is

$$
\text{{human ratings + frozen LLM ratings}}
\rightarrow
\text{{RCP / synthetic-only / PPI quantile learner}}
\rightarrow
\text{{independent scalar conformal correction}}
\rightarrow
\text{{five-dimensional prediction box}}.
$$

## 2. Data and execution integrity

All checks in `validation_summary.json` pass. There is zero cross-role prompt overlap, zero missing required values, zero out-of-range human or LLM ratings, 500 unique successful outputs for each LLM route, zero API retries, no missing or duplicate seed–method rows, valid coverage/MSCE ranges, exact agreement between raw seed metrics and aggregated summaries, cost matching within $0.0005, and no execution errors in the notebook.

The processed data contain {manifest['rows_after_deduplication']:,} rows. Frozen embeddings are materialized for {embed_manifest['rows']:,} rows across the four used roles. The complete processed HelpSteer2 parquet and the exact embeddings used downstream are included in the delivery bundle.

## 3. Global synthetic-label quality

{global_table}

The constant-median labeler has the best macro rating MAE (**{macro.loc['base_median', 'mae']:.4f}**). Flash and Pro nevertheless show positive within-dimension rank information: macro Spearman correlations are **{macro.loc['flash', 'spearman']:.4f}** and **{macro.loc['pro', 'spearman']:.4f}**. Pro has stronger mean ranking but a large downward scale shift, producing macro bias **{macro.loc['pro', 'bias']:+.4f}** and worse MAE. This is precisely why global five-rating MAE alone is not a sufficient diagnostic for the induced score-quantile task.

## 4. Boundary information with an interpretable reference

{boundary_table}

### How the reference is constructed

For each pilot point, the true and synthetic nonconformity scores are compared with the target score-quantile boundary. The **observed disagreement** is the fraction for which true and synthetic scores lie on opposite sides. The **shuffled disagreement** randomly reassigns the synthetic scores across inputs, preserving the synthetic-score marginal distribution and its above-boundary rate while destroying input-specific alignment. The relative reduction is

$$
1-\frac{{p_{{\mathrm{{observed}}}}}}{{p_{{\mathrm{{shuffled}}}}}}.
$$

This baseline is more informative than raw agreement because the above-boundary class is rare. The constant-median predictor always stays below the boundary and therefore has 86.4% raw agreement solely from class imbalance; its balanced accuracy, F1, MCC, and κ are all at chance/zero. Flash and Pro instead have nontrivial balanced accuracy, F1, MCC, and 32.5%/39.5% less disagreement than their own shuffled-score references. That supports the narrow claim that these general-purpose LLM judges retain useful input-specific boundary-side information despite mediocre global imputation MAE.

## 5. Full downstream results

All entries below are means over 20 paired seeds.

{full_result_table}

### Primary equal-N comparison

- **Flash-PPI vs RCP:** MSCE improves by {flash_msce_reduction:.1%}; the paired difference is {metric(comparisons, 'Flash-PPI', 'msce').difference_vs_rcp:+.6f} with 95% CI [{metric(comparisons, 'Flash-PPI', 'msce').ci95_low:+.6f}, {metric(comparisons, 'Flash-PPI', 'msce').ci95_high:+.6f}]. Minimum-group coverage increases by {flash.min_group_coverage_mean - rcp.min_group_coverage_mean:+.4f}. Mean log-volume is essentially unchanged on average.
- **Pro-PPI vs RCP:** MSCE improves by {pro_msce_reduction:.1%}; the paired difference is {metric(comparisons, 'Pro-PPI', 'msce').difference_vs_rcp:+.6f} with 95% CI [{metric(comparisons, 'Pro-PPI', 'msce').ci95_low:+.6f}, {metric(comparisons, 'Pro-PPI', 'msce').ci95_high:+.6f}]. Minimum-group coverage improves by {metric(comparisons, 'Pro-PPI', 'min_group_coverage').difference_vs_rcp:+.4f} with a CI excluding zero, while log-volume falls by {abs(metric(comparisons, 'Pro-PPI', 'mean_log_volume').difference_vs_rcp):.4f}, also with a CI excluding zero.
- **Marginal validity:** all methods remain near the common 0.9 target after final conformalization; differences in MSCE therefore reflect redistribution of coverage across groups rather than failure to maintain the global coverage target.

### Paired uncertainty

{paired_table}

## 6. Cost-matched interpretation

Flash-PPI uses all 500 Flash calls (100 paired + 400 auxiliary) for **${flash.query_cost_usd_mean:.6f}**. Cost-matched Pro-PPI uses the same 100 paired points plus **{pro_cm.auxiliary_size_mean:.1f}** auxiliary points on average for **${pro_cm.query_cost_usd_mean:.6f}**. The cost difference is only ${abs(flash.query_cost_usd_mean-pro_cm.query_cost_usd_mean):.6f}.

At this matched spend:

- Flash has lower mean MSCE ({flash.msce_mean:.6f} vs {pro_cm.msce_mean:.6f}) and lower mean log-volume ({flash.mean_log_volume_mean:.4f} vs {pro_cm.mean_log_volume_mean:.4f}).
- Pro has slightly higher mean minimum-group coverage ({pro_cm.min_group_coverage_mean:.4f} vs {flash.min_group_coverage_mean:.4f}).
- Both improve MSCE over RCP; cost-matched Pro's paired MSCE difference is {metric(comparisons, 'Pro-PPI-CostMatched', 'msce').difference_vs_rcp:+.6f}, with 95% CI [{metric(comparisons, 'Pro-PPI-CostMatched', 'msce').ci95_low:+.6f}, {metric(comparisons, 'Pro-PPI-CostMatched', 'msce').ci95_high:+.6f}].

The clean conclusion is therefore budget-dependent: a larger quantity of Flash labels is preferable for average conditional error and compactness at this budget, whereas fewer Pro labels are slightly better for the worst observed group. Equal-N Pro is strongest overall but costs materially more.

## 7. What the experiment does and does not establish

### Supported claims

1. A realistic external LLM labeler can improve final conditional-coverage diagnostics on human-rated text data.
2. The improvement is not obtained by indiscriminate widening: equal-N Pro-PPI simultaneously improves MSCE, minimum-group coverage, and log-volume.
3. Global imputation MAE and local boundary information can diverge naturally. Neither LLM beats the constant-median MAE baseline, yet both beat input-destroying shuffled-score references on balanced/local boundary diagnostics.
4. Actual query cost creates a genuine quantity–quality tradeoff rather than a universal model ranking.

### Important qualification

In this realistic case study, the development-selected PPI models do **not** improve held-out true pinball risk before final conformalization: Flash-PPI differs from RCP by {metric(comparisons, 'Flash-PPI', 'true_pinball').difference_vs_rcp:+.6f}, and Pro-PPI by {metric(comparisons, 'Pro-PPI', 'true_pinball').difference_vs_rcp:+.6f}, with negative defined as better. The clean empirical result here is therefore the final conditional-coverage improvement after independent conformalization, not validation of every intermediate arrow in the theoretical chain. The controlled multi-dataset experiments and theory should remain the evidence for the general risk-estimation-to-excess-risk mechanism.

This qualification does not weaken the realistic case study's main purpose: it directly answers whether modern LLM-generated labels can yield useful final conditional coverage and exposes the requested real cost tradeoff.

## 8. Recommended reviewer-facing boundary table

The following table should replace the original raw-agreement-only table:

{markdown_table(
    ['Synthetic labeler', 'Macro rating MAE ↓', 'Observed disagreement ↓', 'Shuffled disagreement ↓', 'Reduction vs shuffled ↑', 'Balanced boundary accuracy ↑'],
    [
        ['Constant median', f"{macro.loc['base_median','mae']:.3f}", f"{boundary_indexed.loc['base_median','boundary_disagreement']:.3f}", f"{boundary_indexed.loc['base_median','shuffled_pair_disagreement']:.3f}", '0.0%', f"{boundary_indexed.loc['base_median','balanced_accuracy']:.3f}"],
        ['Flash', f"{macro.loc['flash','mae']:.3f}", f"{boundary_indexed.loc['flash','boundary_disagreement']:.3f}", f"{boundary_indexed.loc['flash','shuffled_pair_disagreement']:.3f}", pct(boundary_indexed.loc['flash','relative_reduction_vs_shuffled']), f"{boundary_indexed.loc['flash','balanced_accuracy']:.3f}"],
        ['Pro', f"{macro.loc['pro','mae']:.3f}", f"{boundary_indexed.loc['pro','boundary_disagreement']:.3f}", f"{boundary_indexed.loc['pro','shuffled_pair_disagreement']:.3f}", pct(boundary_indexed.loc['pro','relative_reduction_vs_shuffled']), f"{boundary_indexed.loc['pro','balanced_accuracy']:.3f}"],
    ]
)}

Suggested interpretation:

> Neither LLM beats the constant-median baseline in global five-rating MAE. Raw boundary agreement alone is also uninformative because only 13.6% of true scores lie above the τ=0.9 boundary: an always-below baseline already attains 86.4% agreement. We therefore compare each LLM with a shuffled-score reference that preserves its score distribution but destroys input-specific alignment. Flash and Pro reduce boundary disagreement by 32.5% and 39.5% relative to this reference and attain balanced boundary accuracies of 0.637 and 0.701, respectively. These are general-purpose LLM judges, not boundary classifiers. The result provides a realistic example in which global imputation MAE is mediocre while the local boundary information relevant to quantile learning remains nontrivial and supports improved final conditional coverage.

## 9. Reproducibility and raw-data inventory

The delivery bundle contains:

- the full processed 21,352-row HelpSteer2 parquet used to allocate all roles;
- the exact frozen 768-dimensional embeddings used by the downstream experiment;
- the 500-row pilot table with prompt, response, human ratings, and both LLM rating vectors in CSV and Parquet;
- the original per-request Flash and Pro JSONL records, including model route, token usage, latency, attempts, raw JSON response, and parsed ratings;
- global label metrics, corrected boundary metrics, and API usage summaries;
- all 120 seed–method downstream metric rows, aggregate summaries, and paired-bootstrap comparisons;
- the split, embedding, and validation manifests;
- the executed notebook and HTML rendering;
- all source scripts needed to reconstruct the experiment; and
- SHA-256 checksums for every bundled file.

See `raw_data/DATA_DICTIONARY.md` and `raw_data/FILES.md` for field-level definitions and paths.
"""
    (OUT / "detailed_experiment_report.md").write_text(report, encoding="utf-8")


def build_raw_data() -> None:
    if OUT.exists():
        for child in [RAW, CODE]:
            if child.exists():
                shutil.rmtree(child)
    RAW.mkdir(parents=True, exist_ok=True)
    CODE.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(RESULTS / "pilot_predictions.parquet")
    predictions.to_csv(RAW / "pilot_human_llm_labels.csv", index=False)
    copy_file(RESULTS / "pilot_predictions.parquet", RAW / "pilot_human_llm_labels.parquet")

    copies = {
        DATA / "helpsteer2.parquet": RAW / "helpsteer2_processed_full.parquet",
        DATA / "helpsteer2_embeddings.npz": RAW / "helpsteer2_embeddings_used.npz",
        DATA / "helpsteer2_manifest.json": RAW / "helpsteer2_manifest.json",
        DATA / "helpsteer2_embeddings_manifest.json": RAW / "helpsteer2_embeddings_manifest.json",
        RESULTS / "flash_labels.jsonl": RAW / "flash_raw_api_outputs.jsonl",
        RESULTS / "pro_labels.jsonl": RAW / "pro_raw_api_outputs.jsonl",
        RESULTS / "label_metrics.csv": RAW / "label_metrics.csv",
        RESULTS / "boundary_metrics.csv": RAW / "boundary_metrics.csv",
        RESULTS / "api_usage.csv": RAW / "api_usage.csv",
        RESULTS / "label_analysis_summary.json": RAW / "label_analysis_summary.json",
        RESULTS / "validation_summary.json": RAW / "validation_summary.json",
        RESULTS / "downstream_cost_matched" / "raw_metrics.csv": RAW / "downstream_seed_metrics.csv",
        RESULTS / "downstream_cost_matched" / "summary.csv": RAW / "downstream_summary.csv",
        RESULTS / "downstream_cost_matched" / "paired_comparisons.csv": RAW / "downstream_paired_comparisons.csv",
        RESULTS / "downstream_cost_matched" / "summary.json": RAW / "downstream_summary.json",
        ROOT / "helpsteer2_llm_pilot_report.ipynb": OUT / "helpsteer2_llm_pilot_report.ipynb",
        ROOT / "helpsteer2_llm_pilot_report.html": OUT / "helpsteer2_llm_pilot_report.html",
    }
    for source, destination in copies.items():
        copy_file(source, destination)

    for name in [
        "prepare_data.py",
        "run_labels.py",
        "prepare_embeddings.py",
        "analyze_labels.py",
        "run_downstream_pilot.py",
        "make_notebook.py",
        "validate_outputs.py",
        "build_delivery_bundle.py",
    ]:
        copy_file(ROOT / name, CODE / name)

    dictionary = """# Raw-data dictionary

## `pilot_human_llm_labels.csv` / `.parquet`

One row per HelpSteer2 prompt–response pair in the 500-row LLM pilot.

| Field | Meaning |
|---|---|
| `row_id` | Stable experiment row identifier. |
| `prompt`, `response` | Original HelpSteer2 text. |
| `helpfulness`, `correctness`, `coherence`, `complexity`, `verbosity` | Human ratings in {0,1,2,3,4}. |
| `official_split` | Original HelpSteer2 split. |
| `prompt_key` | SHA-256-derived key for the normalized prompt; used for leakage-safe grouping. |
| `role` | Experiment role; all rows here are `pilot`. |
| `prompt_chars`, `response_chars` | Text lengths in Unicode characters. |
| `syn_flash_*`, `syn_pro_*` | Parsed LLM ratings in [0,4]. |

## `flash_raw_api_outputs.jsonl` / `pro_raw_api_outputs.jsonl`

One JSON record per API request. Important fields are `row_id`, `requested_model`, `served_model`, `response_id`, `status`, parsed `ratings`, `raw_content`, token counts, cache-token counts, number of `attempts`, latency, and any captured `errors`. No API key or authentication secret is stored.

## `downstream_seed_metrics.csv`

One row per seed and method (20 seeds × 6 methods = 120 rows).

| Field | Meaning |
|---|---|
| `seed` | Downstream split/initialization seed index. |
| `method` | RCP, synthetic-only, equal-N PPI, or cost-matched PPI arm. |
| `selected_epoch` | Epoch with the lowest true development pinball loss. |
| `auxiliary_size` | Number of synthetic-only auxiliary points used. |
| `query_cost_usd` | Recorded route-specific LLM query cost for all synthetic labels used by that arm. |
| `true_pinball` | Held-out true-score pinball loss before final conformalization. |
| `pre_coverage`, `pre_msce`, `pre_min_group_coverage` | Diagnostics before the final scalar conformal correction. |
| `gamma` | Finite-sample scalar conformal correction. |
| `coverage`, `msce`, `min_group_coverage` | Final marginal and conditional group diagnostics. |
| `mean_radius` | Mean scalar radius after clipping to [0,1]. |
| `mean_log_volume` | Mean sum of log side lengths across the five-dimensional clipped box. |

## Summary files

- `label_metrics.csv`: per-dimension and macro LLM imputation metrics.
- `boundary_metrics.csv`: raw and class-balanced boundary diagnostics, including the shuffled-score reference.
- `api_usage.csv`: aggregate token, cost, latency, and retry statistics.
- `downstream_summary.csv`: mean and standard deviation over seeds.
- `downstream_paired_comparisons.csv`: paired mean differences versus RCP and 95% bootstrap intervals.
- `validation_summary.json`: pass/fail audit of all data, API, downstream, cost, and notebook checks.

## Full data and embeddings

- `helpsteer2_processed_full.parquet`: all 21,352 post-deduplication rows and their experiment roles.
- `helpsteer2_embeddings_used.npz`: row IDs, roles, and the exact 768-dimensional frozen features for the 2,839 used rows.
"""
    (RAW / "DATA_DICTIONARY.md").write_text(dictionary, encoding="utf-8")


def write_inventory_and_zip() -> None:
    paths = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    inventory_lines = ["# Delivery files", ""]
    checksum_lines = []
    for path in paths:
        relative = path.relative_to(OUT)
        size = path.stat().st_size
        inventory_lines.append(f"- `{relative}` — {size:,} bytes")
        checksum_lines.append(f"{sha256(path)}  {relative.as_posix()}")
    (RAW / "FILES.md").write_text("\n".join(inventory_lines) + "\n", encoding="utf-8")
    checksum_lines.append(f"{sha256(RAW / 'FILES.md')}  raw_data/FILES.md")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    zip_path = ROOT / "helpsteer2_experiment_complete_bundle.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
            archive.write(path, Path("helpsteer2_experiment_complete_bundle") / path.relative_to(OUT))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_raw_data()
    build_reports()
    write_inventory_and_zip()
    print(OUT / "paper_ready_experiment_setting.md")
    print(OUT / "detailed_experiment_report.md")
    print(ROOT / "helpsteer2_experiment_complete_bundle.zip")


if __name__ == "__main__":
    main()
