# Experiment-to-code map

| Study | Canonical entry | Frozen configuration | Included evidence |
|---|---|---|---|
| Eight-dataset main comparison | `run.py benchmark --suite main` | `configs/benchmark/v11_main.json` | `reference_results/benchmark/` |
| Teacher, pool, trusted budget, ablations | other benchmark suites | remaining benchmark JSON files | same directory, with suite/task/method identifiers |
| Conformal null and information mechanisms | `run.py theory` | `configs/theory/*.json` | `reference_results/theory/` |
| Deployment derivative / three-resource scaling | `run.py followup` | `FollowupConfig` and `build_scaling_plan` in `src/synrcp/followup.py` | `reference_results/theory_followup/` |
| Original six-arm HelpSteer study | `run.py helpsteer` | `llm/helpsteer/run_downstream_pilot.py` | `reference_results/helpsteer/` |
| Profiled-risk/candidate-selection supplement | `run.py supplement --part tabular` | `supplement/config.json` | `reference_results/supplement/` |
| HelpSteer cross-calibrated risk audit | supplement `helpsteer` part | same configuration / runner | same directory |
| Quadratic discrepancy estimator | supplement `statistics` part | same configuration / runner | `statistical.csv`, `variance_ratios.csv` |
| Earlier Civil Comments screen / prompt gate | scripts in `llm/civil_comments/` | arguments and constants in those scripts | source only; not a main-claim evidence block |

`src/synrcp/` is the executed, role-safe engine used for the frozen benchmark. `legacy_baselines/` and `reference_implementations/` are retained to make all original model/objective logic inspectable. They are not alternative implementations silently used by the launcher. In particular, the earlier implementation-plan README that said the final neural runner was unfinished is not reproduced as the current project status: the final runner is `src/synrcp/v11.py`.

The historical table/report scripts under `reference_implementations/evidence_scripts/` expect their old evidence directory layout. They remain source references, while `tools/validate_reference.py` validates the supplied numeric tables in the repository layout. Notebook/report generators in the LLM directories similarly describe the old full-data workflow; no private report snapshots or executed notebook outputs are included.

For manual use of the native engine from this package root:

```bash
PYTHONPATH=src python -m synrcp.cli plan --config configs/benchmark/v11_main.json --output outputs/main_plan.jsonl
PYTHONPATH=src python -m synrcp.cli run-task --plan outputs/main_plan.jsonl --index 0
PYTHONPATH=src python -m synrcp.cli aggregate --output-root outputs/v11_main
```

The 740-row design registry under `plans/` is validated by `src/synrcp_v11/`; it documents the frozen design rather than replacing executable `synrcp.cli` plans. Keep the distinction when scheduling tasks.
