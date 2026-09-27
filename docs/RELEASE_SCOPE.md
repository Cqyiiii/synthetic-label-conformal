# Public release scope

This repository includes the scientific implementation, frozen experiment designs, numerical reference results, eight tabular benchmark inputs, and the numerical HelpSteer2 caches needed by the supported launcher.

The public release changes documentation, package metadata, licensing and data distribution. The main fitting, calibration, selection, statistical estimator, and evaluation implementations, frozen configuration grids, and reference results are retained from the provided research artifact.

No private conversations, local task logs, account identifiers, credentials, provider request identifiers, dependency vendor directories, or model weights are distributed. The HelpSteer2 caches contain numerical ratings, frozen features, public row/group identifiers, and split roles; raw prompt/response text is omitted.

The supported workflows start at `run.py`; older evidence-building and notebook-generation scripts are retained for inspection and may require their historical full-data layout. The Civil Comments screening code is a separate historical study, with its original null/failed-gate interpretation. It is not part of the paper's primary result bundle and its caches are not included.

The code is MIT licensed. External datasets retain their original licenses and attribution, as documented in `THIRD_PARTY_NOTICES.md` and `docs/DATA.md`.

Validation scope and exact executed checks are recorded in `docs/VERIFICATION.json`. A passing package check is not a complete rerun or independent confirmation of every scientific result.
