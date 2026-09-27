# Reproduction boundaries and retained findings

## What is preserved

- The canonical main computational engine, all model/objective functions, theory code, frozen design configurations, seed grids, and main/supplement tests.
- Full numerical reference rows for the benchmark (6,900), resource-scaling follow-up (540), supplemental method/candidate-control tables (420 each), supplemental HelpSteer audit (60), and quadratic-estimator study (32).
- Scientific third-party citations, dataset identities, labeling prompts, model identities, token-count-based historical costs, and dependence limitations.

Package paths, documentation/metadata, console entry points, an explicit device selection argument, and the launcher are adapted. The generic CSV exporter additionally routes `v11` outputs to the real-data branch and reads teacher/budget conditions from their nested location, preventing empty exports or merged conditions. Fitting, calibration, and estimator code are unchanged. The eight tabular datasets are bundled in the frozen input formats. HelpSteer caches remove text and request-level metadata while retaining the numerical inputs used by downstream models. Reference CSVs omit original execution timestamps and source-location fields; numerical metrics are not intentionally changed.

## Main benchmark

The executed learner uses full-batch neural training. Disjoint row-index roles cover preprocessing, teacher training, base prediction, threshold learning, final calibration, grouping, slab selection and evaluation. Imputation/standardization fit only on the declared preprocessing role. Synthetic-pool true labels are not arguments to fitting or selection routines. See `src/synrcp/v11.py` for the exact role prefixes, fixed learner budgets and method mapping.

## Candidate-selection supplement

Four datasets use seeds 0–14 and a common six-candidate bank: training powers 0, 0.5, 1, each evaluated at epochs 50 and 100. Temporary two-fold calibration and selection use the development suffix of the original calibration reservoir, with its true-label cost explicitly counted. The supervised extra-label control trains on learning plus development labels. Selected candidates are saved before final outcome evaluation and are not refit.

The conformal offset uses rank `ceil((m+1)*tau)`, returning infinity if the rank exceeds the calibration sample size. The coverage-moment estimator subtracts sample-covariance traces from the squared corrected mean. Negative estimates are retained. Its exact variance includes all first- and second-order paired/pool components; the first-order term vanishes at zero discrepancy. Positive coupling in the actual simulation is `p*p_syn + 0.85*(min(p,p_syn)-p*p_syn)`, rather than maximal positive coupling.

The supplied results do not support replacing fixed unit PPI with the new coverage selector. Coverage beta=1 has higher mean grouped MSCE on Bike, Diamonds and Transcoding; the SGEMM paired interval includes zero. Do not drop adverse/null results or use the evaluation-only oracle for selection.

## Uncertainty, data quality and interpretation

Paired bootstrap intervals describe variation across overlapping random splits of fixed datasets, not independent population sampling. Multiple comparisons are not corrected. Grouped MSCE, Fourier moments, WSC and ERT are finite-resolution diagnostics, not proofs of pointwise conditional validity.

Diamonds retains 146 exact duplicated X/Y records and 345 duplicate X rows; some full records cross row-index roles. This inherited processing was not changed after observing results. Row-ID separation therefore does not establish entity-level independence.

HelpSteer outer roles use disjoint prompts, but responses within prompts are clustered. The original paired/pool split can split responses to the same prompt across its two samples. Final calibration contains 400 responses from 200 prompts; the official test subset has 1,038 responses from 518 prompts. An ordinary row-iid conformal guarantee is not established for this reconstruction.

The original development set selects neural checkpoints using raw risk. Its risk summaries are adaptive. The additional cross-calibrated test risk uses a fixed prompt-group half split and is diagnostic only. Calibration reduces the positive PPI-vs-RCP risk gap without eliminating it, while grouped coverage error is lower. Risk-value ordering and coverage-gradient norm are not equivalent.

Historical HelpSteer fits and their CPU reconstruction are not numerically identical. Across the three reconstructed arms, maximum absolute differences from archived metrics were approximately 0.012516 for raw pinball, 0.042389 for coverage, 0.004600 for MSCE, and 30 epochs for the selected checkpoint. No unsupported device/version explanation is asserted.

## What this delivery verification does not claim

Packaging tests verify imports, frozen plan counts, formula invariants, calibration ranks, checkpoint behavior, reference-table consistency and representative execution paths. They are not a new execution of all 740 benchmark tasks or a new external replication. No scientific method was retuned during release preparation, and no fresh LLM label or embedding was generated.

Cache reuse is intended only with the frozen configuration. To change a design or rerun from scratch, use a fresh extraction or explicitly separate the old output directory. The supplied reference tables live outside resumable output folders so that they cannot be mistaken for completed local runs.
