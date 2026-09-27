# Third-party data and attribution

The MIT license at the repository root applies to this project's original code. The external datasets and their processed forms retain the terms listed below. Keep these notices with redistributed copies. The data creators do not endorse this project.

## UCI benchmark datasets — CC BY 4.0

All seven UCI dataset pages linked below state [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) licensing ([legal code](https://creativecommons.org/licenses/by/4.0/legalcode)). Their source records were checked on 2026-09-27.

| Included file | Credit and permanent source | Representation in this repository |
| --- | --- | --- |
| `legacy/Datasets/bike_hour.csv` | Hadi Fanaee-T (2013), [Bike Sharing](https://doi.org/10.24432/C5W894) | Hourly `hour.csv`, renamed; 17,379 data rows. |
| `legacy/Datasets/gt_full.csv` | [Gas Turbine CO and NOx Emission Data Set](https://doi.org/10.24432/C5WC95), UCI (2019); study by Heysem Kaya and collaborators | Five yearly files (2011–2015) concatenated in year order, with a stored CSV index; 36,733 rows. The loader drops that index. |
| `legacy/Datasets/naval.csv` | Andrea Coraddu, Luca Oneto, Alessandro Ghio, Stefano Savio, Davide Anguita and Massimo Figari (2014), [Condition Based Maintenance of Naval Propulsion Plants](https://doi.org/10.24432/C5K31K) | Whitespace data converted to CSV with numeric column names; 11,934 rows. |
| `legacy/Datasets/sgemm_product.csv` | Enrique Paredes and Rafael Ballester-Ripoll (2017), [SGEMM GPU Kernel Performance](https://doi.org/10.24432/C5MK70) | Complete 241,600-row file; the experiment loader makes the frozen 50,000-row selection. |
| `legacy/Datasets/super.csv` | Kam Hamidieh (2018), [Superconductivty Data](https://doi.org/10.24432/C53P47) | `train.csv`, renamed; 21,263 rows. |
| `legacy/Datasets/transcoding_measurement.tsv` | Tewodros Deneke (2014), [Online Video Characteristics and Transcoding Time Dataset](https://doi.org/10.24432/C58C9K) | The 68,784-row measurement file, renamed from `transcoding_mesurment.tsv`; not the larger video-catalog file. |
| `legacy/Datasets/WEC_Perth_49.csv` | Mehdi Neshat, Bradley Alexander, Nataliia Sergiienko and Markus Wagner (2020), [Large-scale Wave Energy Farm](https://doi.org/10.24432/C5GG7Q) | The 49-device Perth file, 36,043 rows; not a different location/device-count variant. |

Frozen CSV/TSV values and row order are retained. Model preprocessing is performed by the supplied code using its declared data roles. Repository row counts describe the specific included file, which may differ from a landing page's total across multiple files.

## Diamonds — ggplot2 distribution

`legacy/Datasets/diamonds.csv` contains the 53,940-row [Diamonds dataset distributed with ggplot2](https://ggplot2.tidyverse.org/reference/diamonds.html), credited to the ggplot2 authors and core developer team.

The ggplot2 package declares `MIT + file LICENSE`; the upstream MIT notice is included verbatim in [licenses/ggplot2-MIT.txt](licenses/ggplot2-MIT.txt). The CSV representation was checked against [the upstream R data file](https://github.com/tidyverse/ggplot2/blob/313d0901d955a304ae09b36c13946933b8909eaf/data/diamonds.rda). All cells, columns and row order agree. The representation changes R factor storage to ordinary CSV strings and does not deduplicate or re-order observations.

## HelpSteer2 — CC BY 4.0

The human ratings and associated derived numerical caches originate from [NVIDIA HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2), licensed by its creators under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit NVIDIA and the authors of *HelpSteer2: Open-source dataset for training top-performing reward models* ([paper](https://arxiv.org/abs/2406.08673)).

`llm/helpsteer/data/helpsteer2.parquet` retains 21,352 processed row records after the study's deduplication and prompt-overlap exclusions. Raw prompt/response text is omitted. `helpsteer2_embeddings.npz` supplies 2,839 frozen 768-dimensional feature rows, produced by concatenating normalized prompt and response embeddings using `sentence-transformers/all-MiniLM-L6-v2`. No encoder weights are distributed.

The project's cached synthetic ratings, token counts and derived numerical summaries accompany these inputs. Provider request/response identifiers and account details are excluded. `docs/DATA.md` describes the processing and role assignments; `docs/REPRODUCIBILITY.md` records dependence and reconstruction limits. These caches preserve the underlying dataset attribution and are not represented as unmodified copies of the full upstream text dataset.

## Dependencies and earlier implementations

Python libraries named in the requirements retain their own licenses; dependency source trees and model weights are not vendored here. The historical implementations and scientific references are retained for attribution and inspection. The separate Civil Comments code identifies its own public source; no Civil Comments dataset copy is included in this repository.
