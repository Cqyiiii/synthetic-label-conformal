#!/usr/bin/env python3
"""Build the reader-facing notebook for the executed prompt-calibration gate."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def main() -> None:
    output = Path("prompt_gate/prompt_gate_report.ipynb").resolve()
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3 (ml)",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    notebook["cells"] = [
        markdown(
            """# DeepSeek prompt-calibration gate on Civil Comments

## tl;dr

- **No model/prompt combination passes the pre-registered four-part gate.** The closest is **DeepSeek V4 Flash + interleaved 16-shot ICL**, which passes MAE, bias, and boundary disagreement but not rank correlation.
- The best Flash ICL setting has raw MAE **0.1239**, bias **-0.0132**, Spearman **0.3792**, and boundary disagreement **0.1600**. The thresholds are 0.13, 0.03 in absolute value, 0.45, and 0.16 respectively.
- ICL corrects the zero-shot scale and local score-boundary behavior, but compresses predictions to only **5 unique values**, reducing ranking information. Pro shows the same tradeoff more strongly.
- **Decision:** do not expand Civil Comments to the expensive downstream conformal experiment under the stated gate. A new dataset with a label scale that an LLM can identify more directly is the cleaner next step.
- All 3,000 valid-gate requests succeeded without retries or out-of-range outputs. Estimated cost for the valid interleaved run is **$0.176**. An earlier grouped-example diagnostic was excluded after it exposed an ICL order confound."""
        ),
        markdown(
            """## Context & Methods

This notebook records the go/no-go experiment proposed after the initial zero-shot Civil Comments pilot. It compares three prompts for both DeepSeek routes:

1. zero-shot;
2. 16-shot ICL;
3. 16-shot ICL plus an explicit instruction about the human-vote scale and zero-heavy label distribution.

Prompt selection uses only synthetic-label diagnostics on a new 500-comment validation set. It does **not** use downstream pinball risk, MSCE, WSC, or interval width.

### Key Assumptions

- The 16 prompt-development comments represent historical labeled examples and are excluded from calibration, conformalization, and test use.
- The 500 prompt-validation comments are disjoint from the earlier 400-row base-model split and 500-row pilot, by both row ID and normalized text.
- A fixed 250/250 split is used for fitting and evaluating the diagnostic affine calibrator. Raw prompt metrics use all 500 comments; held-out calibration metrics use only the 250-row evaluation fold.
- Boundary disagreement is evaluated on the held-out fold using one shared true-score quantile model, so prompt comparisons share the same boundary.
- The decision gate is: MAE $\leq 0.13$, absolute bias $\leq 0.03$, Spearman $\geq 0.45$, and boundary disagreement $\leq 0.16$."""
        ),
        code(
            """from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image

cwd = Path.cwd().resolve()
if (cwd / "prompt_gate" / "analysis").exists():
    experiment_root = cwd
elif (cwd / "analysis").exists() and cwd.name == "prompt_gate":
    experiment_root = cwd.parent
else:
    raise FileNotFoundError(f"Cannot locate prompt_gate/analysis from {cwd}")

analysis_dir = experiment_root / "prompt_gate" / "analysis"
data_dir = experiment_root / "prompt_gate" / "data"
summary = pd.read_csv(analysis_dir / "prompt_gate_summary.csv")
quality = pd.read_csv(analysis_dir / "prompt_gate_data_quality.csv")
comparisons = pd.read_csv(analysis_dir / "prompt_variant_paired_comparisons.csv")
assignments = pd.read_parquet(data_dir / "prompt_gate_assignments.parquet")
decision = json.loads((analysis_dir / "prompt_gate_decision.json").read_text())
manifest = json.loads((data_dir / "prompt_gate_manifest.json").read_text())"""
        ),
        markdown("## Data\n\n### 1. Verify the sample assignment and leakage checks"),
        code(
            """quality_checks = manifest["quality_checks"]
sample_check = pd.DataFrame({
    "check": [
        "Prompt-development rows", "Prompt-validation rows", "Unique row IDs",
        "Unique normalized text", "Historical row overlap", "Historical text overlap",
        "Missing required values", "Out-of-range true labels"
    ],
    "result": [
        quality_checks["prompt_dev_rows"], quality_checks["prompt_validation_rows"],
        quality_checks["unique_row_ids"], quality_checks["unique_normalized_text"],
        quality_checks["overlap_with_historical_row_ids"],
        quality_checks["overlap_with_historical_text"],
        quality_checks["missing_required_values"], quality_checks["toxicity_out_of_range"]
    ]
})
display(sample_check)
display(pd.DataFrame({
    "prompt_dev_bin_count": quality_checks["dev_bin_counts"],
    "validation_bin_count": quality_checks["validation_bin_counts"]
}).fillna(0).astype(int))"""
        ),
        markdown(
            """The assignment checks pass: the 16 development examples and 500 validation rows are unique, mutually exclusive, and have no row-ID or normalized-text overlap with the earlier experiment. The validation sample preserves the naturally zero-heavy label distribution rather than balancing toxicity classes."""
        ),
        markdown("### 2. Verify API completion and output validity"),
        code(
            """display(quality)
assert (quality["successful_unique_rows"] == 500).all()
assert (quality["error_records"] == 0).all()
assert (quality["out_of_range_successes"] == 0).all()
assert (quality["duplicate_success_records"] == 0).all()"""
        ),
        markdown(
            """All six model/prompt combinations contain exactly 500 valid predictions. There are no failed requests, duplicate successful rows, invalid JSON values, or outputs outside $[0,1]$. Thinking mode was disabled in every request; the raw JSONL files preserve served routes, token usage, latency, and retry counts."""
        ),
        markdown("## Results\n\n### 3. Compare the prompt-calibration metrics"),
        code(
            """view_columns = [
    "model", "prompt_variant", "raw_mae", "raw_bias", "raw_spearman",
    "boundary_disagreement", "n_unique_predictions", "heldout_calibrated_r2",
    "estimated_cost_usd", "gate_pass_count", "gate_pass_all"
]
display(summary[view_columns].sort_values(["model", "prompt_variant"]).style.format({
    "raw_mae": "{:.4f}", "raw_bias": "{:+.4f}", "raw_spearman": "{:.4f}",
    "boundary_disagreement": "{:.3f}", "heldout_calibrated_r2": "{:.3f}",
    "estimated_cost_usd": "${:.4f}"
}))"""
        ),
        code("display(Image(filename=str(analysis_dir / 'prompt_gate_metrics.png')))"),
        markdown(
            """The tradeoff is consistent across routes. Zero-shot retains more ranking information but overpredicts the human-vote fraction and has worse boundary disagreement. ICL moves the prediction scale downward and improves local boundary behavior, but maps most comments to a handful of coarse scores. The explicit distributional instruction does not restore ranking resolution beyond ordinary ICL."""
        ),
        markdown("### 4. Inspect paired changes from zero-shot"),
        code(
            """display(comparisons.style.format({
    "candidate_minus_reference": "{:+.4f}", "ci95_low": "{:+.4f}", "ci95_high": "{:+.4f}"
}))"""
        ),
        markdown(
            """For Flash, interleaved ICL reduces MAE relative to zero-shot; for Pro, both ICL variants substantially reduce MAE and boundary disagreement. The rank-correlation changes have wide paired intervals and do not establish an improvement. This is not merely a global accuracy failure: the main limitation is that the prompt cannot simultaneously deliver a well-calibrated soft-label scale and preserve fine ordering."""
        ),
        markdown("### 5. Apply the go/no-go rule without downstream selection"),
        code(
            """display(pd.DataFrame(decision["gate_passers"]))
print(json.dumps(decision, indent=2))"""
        ),
        markdown(
            """## Takeaways

1. **ICL was worth testing, but it does not clear the gate.** The closest setting passes three of four criteria, missing Spearman by a material margin.
2. **The first grouped-example run is not used.** Its label-blocked example order placed zero-score demonstrations at the end and introduced a recency confound. The reported run uses a fixed seeded interleaving; the excluded raw files remain available for audit.
3. **Flash is the stronger operational route for this experiment.** Its best setting is closer to the full gate and is materially cheaper than Pro. This supports treating Flash and Pro as two current API routes, not assuming that the higher-price preview route is stronger.
4. **Do not scale Civil Comments yet.** The clean next move is a dataset whose continuous/ordinal human label has a directly legible rating scale for an LLM (for example, review stars or human response-quality ratings). That avoids spending the larger experiment budget on a surrogate that cannot preserve both calibration and ranking.
5. If Civil Comments must be retained, the next defensible method would be a separately specified score-calibration model evaluated on another untouched validation set—not further prompt tuning on these same 500 rows."""
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
