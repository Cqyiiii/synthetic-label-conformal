# Data provenance and preparation

## Tabular benchmark

The eight tabular datasets are included in `legacy/Datasets/` in the exact formats used by the frozen experiments. Their canonical public sources are listed below. `tabular_data.json` records the exact filenames, row counts, and ordered columns. `tools/prepare_tabular_data.py` can validate the bundled inputs in place or import separately obtained copies; it rejects schema or row-count mismatches. These lightweight checks do not certify equality of every value in an arbitrary replacement copy.

```bash
python tools/prepare_tabular_data.py --from-local legacy/Datasets
```

| Dataset | Canonical source | Expected file and preparation |
|---|---|---|
| Bike | [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | `hour.csv` → `bike_hour.csv`; 17,379 hourly rows |
| Diamonds | [ggplot2 diamonds](https://ggplot2.tidyverse.org/reference/diamonds.html) | `diamonds.csv`, 53,940 rows, no added row-index column |
| Gas Turbine | [UCI Gas Turbine CO and NOx](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set) | concatenate `gt_2011.csv` through `gt_2015.csv` in year order; save `gt_full.csv` with an index |
| Naval | [UCI Naval Propulsion Plants](https://archive.ics.uci.edu/dataset/316/condition+based+maintenance+of+naval+propulsion+plants) | whitespace `data.txt`, 18 columns; save `naval.csv` with numeric column names and no row index |
| SGEMM | [UCI SGEMM GPU Kernel Performance](https://archive.ics.uci.edu/dataset/440/sgemm+gpu+kernel+performance) | `sgemm_product.csv`, all 241,600 rows |
| Superconductivity | [UCI Superconductivty Data](https://archive.ics.uci.edu/dataset/464/superconductivty+data) | `train.csv` → `super.csv`; 21,263 rows |
| Transcoding | [UCI Online Video Characteristics and Transcoding Time](https://archive.ics.uci.edu/dataset/335/online+video+characteristics+and+transcoding+time+dataset) | `transcoding_measurement.tsv`; 68,784 rows; renamed from the upstream `transcoding_mesurment.tsv` spelling |
| WEC | [UCI Large-scale Wave Energy Farm](https://archive.ics.uci.edu/dataset/882/large-scale+wave+energy+farm) | specifically `WEC_Perth_49.csv`, 36,043 rows, not a different farm or the 100-device variant |

No download is needed for the bundled-data workflow. To rebuild the inputs from the public archives, extract the required files into one directory, follow the naming and conversion rules above, and run:

```bash
python tools/prepare_tabular_data.py --from-local /path/to/public_tables
```

The importer also accepts the raw Bike/Superconductivity filenames, Naval `data.txt`, and the five Gas Turbine year files. To export Diamonds from its documented R source:

```r
write.csv(ggplot2::diamonds, "diamonds.csv", row.names = FALSE)
```

Do not sort or deduplicate rows for exact reproduction. Do not add an extra CSV index except for `gt_full.csv`, whose loader explicitly drops it. The recorded source schema is the authority if a public repository is updated. SGEMM is deterministically capped to 50,000 rows inside the executed loader using seed 2026; do not pre-subsample it.

The seven UCI sources identify their data as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Dataset credits, permanent source links, and processing changes are documented in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The code's MIT license does not replace these dataset terms.

Diamonds is distributed as a CSV representation of the data included in the MIT-licensed ggplot2 distribution. The bundled rows and every cell were compared with `data/diamonds.rda` at ggplot2 revision `313d0901d955a304ae09b36c13946933b8909eaf`; values, column order, and row order agree. The upstream license is retained in [licenses/ggplot2-MIT.txt](../licenses/ggplot2-MIT.txt). No deduplication was performed.

## HelpSteer2

The cached input is derived from [NVIDIA HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2), distributed under CC BY 4.0 by its dataset authors. The original preparation code is in `llm/helpsteer/prepare_data.py`. It removes six duplicated prompt/response rows, excludes one prompt shared across official splits, and assigns prompt-separated outer roles.

Included caches preserve row IDs, prompt-group keys, role assignments, five human ratings, text lengths, paired synthetic ratings, token counts, and frozen features. Prompt/response text is omitted. The joined downstream arrays are unchanged by this column removal. The cache contains 2,839 feature rows with 768 dimensions, formed from concatenated normalized prompt and response embeddings using `sentence-transformers/all-MiniLM-L6-v2`, length limit 256. Model weights are not included or downloaded by the launcher.

The supplied Flash/Pro synthetic labels were already generated for the original study. The literal historical labeling prompts and model identifiers are retained in `run_labels.py`. No API key, request identifier, raw provider response envelope, or account information is distributed. The supplied token counts preserve the original cost-matched comparison; hard-coded historical prices in that analysis are study inputs, not claims about current API pricing.

Preparing fresh embeddings/labels requires obtaining public text and following the original scripts with separately configured dependencies and service access. It is not necessary for the cached reproduction path and was not done during release preparation.

## Civil Comments

`llm/civil_comments/prepare_data.py` identifies its public dataset source and frozen sampling rules. All original classification and prompt-gate code is included. Its raw text, generated-label caches, and encoder/feature artifacts are not bundled. This earlier screening workflow is separate from the main eight-dataset benchmark and HelpSteer claim. Retain the original null/failed-gate interpretation when inspecting its source; including it does not promote it to a successful confirmatory experiment.
