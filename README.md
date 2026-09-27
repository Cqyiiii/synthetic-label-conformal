<div align="center">

<h1>How Synthetic Labels Improve<br>Conformal Prediction</h1>

<p><strong>A Perspective on Conditional Coverage</strong></p>

<p>
  Qianyi Chen &nbsp;·&nbsp; Bo Li<br>
  <sub>School of Economics and Management, Tsinghua University</sub>
</p>

<p>
  <a href="requirements.txt"><img src="docs/assets/badge-python.svg" alt="Python 3.11"></a>
  <a href="LICENSE"><img src="docs/assets/badge-license.svg" alt="Code license: MIT"></a>
  <a href="https://github.com/Cqyiiii/synthetic-label-conformal/tree/v1.0.0"><img src="docs/assets/badge-release.svg" alt="Release v1.0.0"></a>
  <a href="#data"><img src="docs/assets/badge-data.svg" alt="Eight datasets included"></a>
</p>

<p>
  <a href="#quick-start">Quick start</a> &nbsp;·&nbsp;
  <a href="#experiments">Experiments</a> &nbsp;·&nbsp;
  <a href="#data">Data</a> &nbsp;·&nbsp;
  <a href="#documentation">Documentation</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

</div>

---

Research code and data for studying how synthetic labels can improve conditional coverage through prediction-powered quantile learning and conformal calibration. The repository includes the tabular benchmarks, theory diagnostics, cached HelpSteer2 study, and post-conformal selection experiments, with frozen configurations and numerical reference results.

<table>
  <tr>
    <td align="center"><strong>8</strong><br><sub>Tabular datasets</sub></td>
    <td align="center"><strong>740</strong><br><sub>Benchmark tasks</sub></td>
    <td align="center"><strong>6,900</strong><br><sub>Reference task–method rows</sub></td>
    <td align="center"><strong>2,839 × 768</strong><br><sub>Cached HelpSteer features</sub></td>
  </tr>
</table>

> **Ready to reproduce:** datasets, cached ratings, features and reference results are included. After installing dependencies, the supplied-data workflows run offline on CPU, without API credentials, LLM requests or model-weight downloads.

## Quick start

### 1. Install

Use **Python 3.11** with the versions in [requirements.txt](requirements.txt).

```bash
git clone https://github.com/Cqyiiii/synthetic-label-conformal.git
cd synthetic-label-conformal
python -m pip install -r requirements.txt
```

### 2. Check the supplied code and data

```bash
python -m pytest -q
python tools/prepare_tabular_data.py --from-local legacy/Datasets
python run.py plan
python run.py check-results
```

These checks validate the package, input schemas, frozen design and supplied reference tables. They do not rerun the full benchmark.

### 3. Run one benchmark task

```bash
python run.py benchmark --suite main --smoke
```

The smoke run executes one task with the original method settings. The complete benchmark is substantially more expensive; expand the experiment commands below when ready.

<details>
<summary><strong>Conda setup and optional console commands</strong></summary>

Instead of installing dependencies into an existing environment:

```bash
conda env create -f environment.yml
conda activate synthetic-conformal
```

The Python package and Conda environment are named `synthetic-conformal`. The repository directory is `synthetic-label-conformal`.

Editable installation is optional for `run.py`, which resolves the included source tree directly and also works by absolute path from another working directory. Install the console commands before using generated SLURM jobs:

```bash
python -m pip install -e .
```

</details>

## Experiments

| Study | Frozen scope | Configuration or evidence |
| :--- | :--- | :--- |
| Main comparison | 8 datasets × 30 common splits; 10 methods | [Main configuration](configs/benchmark/v11_main.json) |
| Teacher, pool, label budget and ablations | 740 tasks / 6,900 method rows across all benchmark suites | [Benchmark designs](configs/benchmark/) |
| Theory diagnostics | 260 tasks | [Theory configurations](configs/theory/) |
| Resource scaling | 270 tasks / 540 method rows, plus a population derivative audit | [Follow-up evidence](reference_results/theory_followup/) |
| HelpSteer2 | Six arms using cached human/synthetic ratings and features | [HelpSteer evidence](reference_results/helpsteer/) |
| Post-conformal selection | 60 tabular tasks; six neural candidates per task | [Supplement configuration](supplement/config.json) |
| Coverage-discrepancy estimator | 32 settings × 100,000 Monte Carlo repetitions | [Statistical evidence](reference_results/supplement/statistical.csv) |

<details>
<summary><strong>Tabular benchmark — all suites and individual runs</strong></summary>

The ten methods are Split, SPPI, NNM, CQR, CQR-PPI, RCP, RCP-PTFT, RCP-PPI, RCP-PPI-CV, and RCP-Aug. The benchmark includes teacher-quality, synthetic-pool-size, trusted-label-budget, and genuine/permuted-label ablations.

```bash
# Full frozen benchmark, sequential and resumable.
python run.py benchmark --suite all

# Individual suites.
python run.py benchmark --suite teacher_quality
python run.py benchmark --suite pool_size
python run.py benchmark --suite label_budget
python run.py benchmark --suite ablations
```

Tasks use one numerical-library thread. Independent tasks can be scheduled through `python -m synrcp.cli` after installing the project or setting `PYTHONPATH=src`. The separate `plans/` files preserve the 740-task design registry; executable configurations are under `configs/benchmark/`.

Regenerated tables are in each suite's `aggregate/real_data_rows.csv`, with one row per task, method and diagnostic group count. This long-form export differs from the frozen `task_method_rows.csv`; compare the same task, method, metric and group count.

</details>

<details>
<summary><strong>Theory diagnostics and resource-scaling follow-up</strong></summary>

```bash
python run.py theory --suite conformal_null --smoke
python run.py theory --suite all
python run.py followup
python tools/validate_theory_followup.py outputs/theory_followup
```

The follow-up contains the population derivative audit and the 270-task resource-scaling design.

</details>

<details>
<summary><strong>HelpSteer2 — cached realistic-label study</strong></summary>

```bash
# Original six-arm study, including the cost-matched arm.
python run.py helpsteer --smoke
python run.py helpsteer
```

The launcher uses supplied ratings, token-count records and 2,839 cached 768-dimensional features. It makes no LLM request and performs no encoder inference. The CPU setting matches the recorded supplement reconstruction; floating-point differences across installations remain possible.

Original preparation, prompts, labeling and notebook-generation source remains under `llm/helpsteer/` and `llm/civil_comments/`. These historical workflows use optional dependencies from `requirements-optional.txt` and are not invoked by the reproduction launcher. The separately reported Civil Comments screening study is source only; its caches are not included. See the [experiment map](docs/EXPERIMENTS.md) before using historical entry points.

</details>

<details>
<summary><strong>Post-conformal selection, risk audit and estimator study</strong></summary>

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

The tabular supplement uses four datasets × 15 seeds, six neural candidates per task, four selectors, fixed supervised/PPI baselines and an extra-label supervised control. Three trajectories use powers 0, 0.5 and 1, with checkpoints at epochs 50 and 100. The HelpSteer risk audit uses 20 seeds × three arms. The estimator study uses 32 settings with 100,000 Monte Carlo repetitions per setting.

Analysis, audits, figures and display tables require the preceding runs. They do not silently substitute supplied reference results for missing executions.

</details>

> **Output and cache handling.** Benchmark/theory smoke outputs are isolated under `outputs/smoke_*`. The tabular supplement smoke result is a genuine first task reused by the full supplement. If you change a design, use a fresh extraction or a new output directory so that old caches are not mixed with changed configurations.

## Data

All eight benchmark tables are included in their frozen input formats under [legacy/Datasets/](legacy/Datasets/). Preserve their row order for exact reproduction.

| Dataset | Included rows | Input file |
| :--- | ---: | :--- |
| Bike Sharing | 17,379 | [bike_hour.csv](legacy/Datasets/bike_hour.csv) |
| Diamonds | 53,940 | [diamonds.csv](legacy/Datasets/diamonds.csv) |
| Gas Turbine | 36,733 | [gt_full.csv](legacy/Datasets/gt_full.csv) |
| Naval Propulsion | 11,934 | [naval.csv](legacy/Datasets/naval.csv) |
| SGEMM | 241,600 | [sgemm_product.csv](legacy/Datasets/sgemm_product.csv) |
| Superconductivity | 21,263 | [super.csv](legacy/Datasets/super.csv) |
| Video Transcoding | 68,784 | [transcoding_measurement.tsv](legacy/Datasets/transcoding_measurement.tsv) |
| Wave Energy | 36,043 | [WEC_Perth_49.csv](legacy/Datasets/WEC_Perth_49.csv) |

SGEMM is included in full; the engine makes its frozen 50,000-row selection. HelpSteer2 is supplied as processed numerical records, cached synthetic ratings and features; raw prompt/response text and API response identifiers are excluded.

See [data provenance and preparation](docs/DATA.md) for sources, schemas and processing, and [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and data licenses.

## Documentation

| Start here to… | Read |
| :--- | :--- |
| Map paper experiments to code and evidence | [Experiment guide](docs/EXPERIMENTS.md) |
| Understand data provenance and formats | [Data guide](docs/DATA.md) |
| Check assumptions and reconstruction limits | [Reproduction boundaries](docs/REPRODUCIBILITY.md) |
| Inspect exactly what is included | [Release scope](docs/RELEASE_SCOPE.md) |
| Review the recorded package checks | [Verification JSON](docs/VERIFICATION.json) · [Standalone HTML report](docs/release-validation.html) |
| Review AI assistance | [AI assistance note](docs/AI_USE.md) |

<details>
<summary><strong>Repository map and historical source scope</strong></summary>

| Directory | Role |
| :--- | :--- |
| `src/synrcp/` | Executed engine, neural learners, calibration, metrics, simulations and resource follow-up |
| `src/synrcp_v11/`, `plans/` | Frozen design registry and validation |
| `configs/benchmark/`, `configs/theory/` | Complete executed parameter grids |
| `supplement/` | Selection module, runners, aggregation, diagnostics and tests |
| `legacy/Datasets/` | Eight included benchmark datasets |
| `llm/helpsteer/` | Realistic-label study, prompts and numerical caches |
| `llm/civil_comments/` | Earlier classification/prompt-gate study source |
| `legacy_baselines/` | Original baseline implementation source |
| `reference_implementations/` | Earlier audited implementations and evidence-building scripts |
| `reference_results/` | Supplied numerical evidence, separate from regenerated output |
| `tools/` | Data import, reference validation and retained study validators |

The scientific objectives, candidate sets, seeds and numerical reference results are preserved. Internal `v11` names identify frozen experiment schemas. `run.py` is the supported reproduction entry point; historical evidence-building scripts may expect an earlier directory layout.

The source covers experimental computation, analysis, data preparation and plotting. Private report-app snapshots, environment-specific delivery/export wrappers and generated notebooks are excluded; they contain no additional fitting or statistical estimator. The supplemental report's numerical display-table transformations are preserved in `supplement/display_tables.py`.

No model weights, dependency vendor trees, credentials, raw provider response envelopes, private chat transcripts, operating-system metadata or original task/session logs are included.

</details>

## Reproducibility and limits

> **Read the reference results with their scope.** Release verification is not a fresh rerun of the entire 740-task benchmark. The additional coverage selectors do **not** consistently outperform fixed unit PPI, and the HelpSteer risk gap shrinks after cross-calibration but persists. Negative/null results and all supplied method rows remain available in `reference_results/`.

Interpretation must account for overlapping random splits, unadjusted paired intervals, finite-resolution conditional-coverage diagnostics, exact duplicate records in Diamonds, prompt clustering in HelpSteer, and non-identical numerical reconstruction of historical neural fits. The estimator simulation's positive-coupling formula uses 85% of the distance to maximal coupling. See the [full reproducibility notes](docs/REPRODUCIBILITY.md).

## Citation

If you use this work, please cite the paper and the original dataset sources. Until a permanent paper identifier is available:

```bibtex
@misc{chen2026syntheticconformal,
  title = {How Synthetic Labels Improve Conformal Prediction: A Perspective on Conditional Coverage},
  author = {Chen, Qianyi and Li, Bo},
  year = {2026},
  url = {https://github.com/Cqyiiii/synthetic-label-conformal}
}
```

Machine-readable software citation metadata is available in [CITATION.cff](CITATION.cff).

## License

The project's original code is released under the [MIT License](LICENSE). Third-party datasets retain their original licenses; the code license does not replace them. Keep the [dataset credits and license notices](THIRD_PARTY_NOTICES.md) with redistributed data.
