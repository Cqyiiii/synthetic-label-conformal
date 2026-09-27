# How Synthetic Labels Improve Conformal Prediction

Code and data for **How Synthetic Labels Improve Conformal Prediction: A Perspective on Conditional Coverage**, by **Qianyi Chen and Bo Li**, School of Economics and Management, Tsinghua University.

This repository contains prediction-powered quantile learning, conformal calibration, eight regression benchmarks, controlled theory diagnostics, the cached HelpSteer2 study, and additional post-conformal selection experiments. The eight benchmark datasets, cached human/synthetic ratings and features, frozen configurations, and numerical reference results are included.

The scientific objectives, candidate sets, random seeds, and numerical reference results are preserved. A common launcher and repository-relative paths support reproduction without private files. Internal `v11` names identify frozen experiment schemas. Raw provider response envelopes, credentials, local execution metadata, and model weights are excluded.

## Install and quick checks

Use Python 3.11 with the versions in `requirements.txt`. A GPU is not required. Clone the repository and run:

```bash
git clone https://github.com/Cqyiiii/synthetic-conformal.git
cd synthetic-conformal
python -m pip install -r requirements.txt
python -m pytest -q
python tools/prepare_tabular_data.py --from-local legacy/Datasets
python run.py plan
python run.py check-results
```

Alternatively, create the environment with `conda env create -f environment.yml` and activate `synthetic-conformal`. Editable installation is optional; `run.py` resolves the included source tree directly and also works when launched by absolute path from a different working directory.

The supplied-data workflows work offline after installing dependencies. They require no API credentials, LLM requests, encoder inference, or model-weight downloads. The tabular files are already in `legacy/Datasets/`, the paths expected by the frozen engine. See [data provenance and licenses](docs/DATA.md), [third-party notices](THIRD_PARTY_NOTICES.md), and [reproduction boundaries](docs/REPRODUCIBILITY.md).

## Main experiments

The frozen benchmark contains **740 tasks and 6,900 task–method rows**, including the eight-dataset main comparison, teacher-quality, synthetic-pool-size, trusted-label-budget, and genuine/permuted-label ablations. The main comparison uses 30 common splits per dataset. Its ten methods are Split, SPPI, NNM, CQR, CQR-PPI, RCP, RCP-PTFT, RCP-PPI, RCP-PPI-CV, and RCP-Aug.

```bash
# One task, all original method settings; a packaging smoke check.
python run.py benchmark --suite main --smoke

# Full frozen benchmark, sequential and resumable.
python run.py benchmark --suite all

# Optional individual suites:
python run.py benchmark --suite teacher_quality
python run.py benchmark --suite pool_size
python run.py benchmark --suite label_budget
python run.py benchmark --suite ablations
```

The full benchmark is substantially more expensive than a smoke check. Tasks use one numerical-library thread; independent tasks can also be scheduled through `python -m synrcp.cli` after installing the project or setting `PYTHONPATH=src`. The separate `plans/` files preserve the 740-task design registry; executable configurations are under `configs/benchmark/`.

Install the optional console commands with `python -m pip install -e .` before using generated SLURM jobs. Regenerated benchmark tables are in each suite's `aggregate/real_data_rows.csv` (one row per task, method, and diagnostic group count). This long-form export differs in layout from the frozen `task_method_rows.csv`; compare the same task, method, metric, and group count.

## Theory and realistic-label studies

```bash
python run.py theory --suite conformal_null --smoke
python run.py theory --suite all
python run.py followup
python tools/validate_theory_followup.py outputs/theory_followup

# Original six-arm HelpSteer study, including the cost-matched arm.
python run.py helpsteer --smoke
python run.py helpsteer
```

The follow-up contains a population derivative audit and 270 resource-scaling tasks / 540 method rows. The HelpSteer launcher uses the supplied ratings, token-count records, and 2,839 cached 768-dimensional features. It makes no LLM request and performs no encoder inference. The CPU setting matches the recorded supplement reconstruction; floating-point differences across installations remain possible.

`llm/helpsteer/` and `llm/civil_comments/` also retain the original data preparation, prompts, labeling, analysis, and notebook-generation source. These are historical workflow components, including the separately reported Civil Comments screening study. The optional generation dependencies are in `requirements-optional.txt`; generation scripts are not invoked by the reproduction launcher. The raw public text and API response identifiers are excluded from the supplied caches. See [experiment map](docs/EXPERIMENTS.md) before using historical entry points.

## Additional post-conformal experiments

```bash
python run.py supplement --part tabular --smoke --workers 1
python run.py supplement --part tabular --workers 3
python run.py supplement --part helpsteer
python run.py supplement --part statistics
python run.py supplement --part analyze
python run.py supplement --part audit
python run.py supplement --part figures
python supplement/display_tables.py
```

The tabular supplement uses four datasets × 15 seeds, six neural candidates per task, four selectors, fixed supervised/PPI baselines, and an extra-label supervised control. Its three trajectories use powers 0, 0.5, and 1, with checkpoints at epochs 50 and 100. The HelpSteer risk audit uses 20 seeds × three arms. The estimator study contains 32 settings with 100,000 Monte Carlo repetitions per setting. The last three commands require completed preceding runs; they do not silently substitute supplied reference results for missing executions.

The tabular smoke result is a genuine first task and is reused by the full supplement; it is not a new scientific comparison. Benchmark/theory smoke outputs are isolated under `outputs/smoke_*`. For a changed design, use a fresh extraction or a new output directory: old caches must not be mixed with changed configurations.

## Code and evidence map

| Directory | Role |
|---|---|
| `src/synrcp/` | Executed main engine, neural learners, conformal calibration, metrics, simulations, resource follow-up |
| `src/synrcp_v11/`, `plans/` | Frozen design registry and validation |
| `configs/benchmark/`, `configs/theory/` | Complete executed parameter grids |
| `supplement/` | New selection module, runners, aggregation, diagnostics, and tests |
| `legacy/Datasets/` | Eight included benchmark datasets in the frozen input formats |
| `llm/helpsteer/` | Original realistic-label study, prompts, and cached numeric inputs |
| `llm/civil_comments/` | Earlier classification/prompt-gate study source |
| `legacy_baselines/` | Original baseline implementation source |
| `reference_implementations/` | Earlier audited implementations and historical evidence-building scripts |
| `reference_results/` | Supplied numeric evidence, distinct from regenerated output |
| `tools/` | Data import, reference validation, and retained study validators |

Historical evidence-building scripts reference the earlier evidence layout and are provided for completeness, not as the current reproduction entry point. The canonical paths above and `run.py` specify the supported workflow. No model weights, dependency vendor trees, private chat transcripts, operating-system metadata files, or original task/session logs are included.

The source scope covers experimental computation, analysis, data preparation, and plotting. Private report-app snapshots, its environment-specific delivery/export wrappers, and generated notebooks are excluded. They contain no additional fitting or statistical estimator. Display-table transformations from the supplemental report are preserved separately in `supplement/display_tables.py`.

## Interpretation and limits

Reference results are included for inspection. Packaging verification is not a fresh rerun of the entire 740-task benchmark. The new coverage selectors do **not** consistently outperform fixed unit PPI; the HelpSteer risk gap shrinks after cross-calibration but persists. Negative/null results and all supplied method rows remain in `reference_results/`.

Important statistical boundaries include overlapping random splits, unadjusted paired intervals, finite-resolution conditional-coverage diagnostics, exact duplicate records in Diamonds, prompt clustering in HelpSteer, and non-identical numerical reconstruction of historical neural fits. The explicit positive-coupling formula in the estimator simulation uses 85% of the distance to maximal coupling. See [reproducibility details](docs/REPRODUCIBILITY.md).

See the [release scope](docs/RELEASE_SCOPE.md), [verification record](docs/VERIFICATION.json), [standalone release report](docs/release-validation.html), and [AI assistance note](docs/AI_USE.md). Dataset/provider names and third-party citations are retained as scientific attribution.

## License and citation

The project's original code is released under the [MIT License](LICENSE). Third-party datasets keep their original licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The MIT code license does not replace those data licenses.

If you use this work, please cite the paper and the original dataset sources. Until a permanent paper identifier is available:

```bibtex
@misc{chen2026syntheticconformal,
  title = {How Synthetic Labels Improve Conformal Prediction: A Perspective on Conditional Coverage},
  author = {Chen, Qianyi and Li, Bo},
  year = {2026},
  url = {https://github.com/Cqyiiii/synthetic-conformal}
}
```

Machine-readable software citation metadata is provided in [CITATION.cff](CITATION.cff).
