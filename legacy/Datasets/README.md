The eight benchmark tables are included in this directory. Keep their row order and filenames unchanged: the frozen engine and split seeds refer to these inputs.

Run `python tools/prepare_tabular_data.py --from-local legacy/Datasets` from the repository root to check all eight schemas and row counts without copying files. Run `python run.py check-results` to check the cached reference inputs and evidence. See `docs/DATA.md`, `docs/tabular_data.json`, and `THIRD_PARTY_NOTICES.md` for data identities, processing details, attribution, and licenses.

To replace the inputs with separately obtained copies in the documented formats, run `python tools/prepare_tabular_data.py --from-local /path/to/public_tables`.
