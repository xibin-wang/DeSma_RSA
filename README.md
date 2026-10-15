# Time-Resolved EEG Representational Similarity Analysis (RSA)

Python pipeline for **time-resolved representational similarity analysis (RSA)** of epoched EEGLAB EEG data. The project compares six semantic conditions across two stimulus-onset-asynchrony (SOA) codes, uses independent stimulus-block partitions to estimate cross-validated neural distances, and produces participant-level results, group plots, quality-control records, and permutation-based statistical summaries. An optional follow-up examines smoothing sensitivity for H2/H4 and explores temporal alignment of H2 curves between age groups.

## Project files

| File | Purpose |
| --- | --- |
| `constant_variable.py` | Paths, condition/event mappings, model parameters, and sensitivity-analysis settings. |
| `data_load.py` | EEGLAB event parsing, epoch metadata, block assignments, and partition selection. |
| `rsa.py` | Participant-level, time-resolved RDMs and theoretical contrasts. |
| `visualisation.py` | Related-condition figures, group summaries, and statistical tests. |
| `main.py` | Primary analysis entry point: data loading, QC, RSA, and output generation. |
| `sensitivity_h2_h4.py` | Optional post-hoc H2/H4 smoothing and exploratory temporal-shift analysis. |

> **Filename note:** Some shared copies have `(2)` appended to their names (for example, `main(2).py`). Rename these to the filenames above before running: the Python imports require `main.py`, `data_load.py`, `rsa.py`, `visualisation.py`, and `constant_variable.py`.

## Requirements

- Python 3.10+ recommended
- NumPy
- pandas
- SciPy
- MNE-Python
- Matplotlib

Install dependencies in a virtual environment:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install numpy pandas scipy mne matplotlib
```

## Input data

The primary pipeline expects **epoched EEGLAB `.set` datasets** with readable `event`, `urevent`, and `trials` information. Additional EEGLAB-associated data files, if referenced by a `.set` file, must remain accessible.

Subjects are normally specified through `meta_data.csv` with these mandatory columns:

```csv
subject,path,group
sub-001,EEG_data/sub-001.set,GJ
sub-002,EEG_data/sub-002.set,GA
```

- `subject`: unique participant identifier, also used in output filenames.
- `path`: path to the `.set` file. Relative paths are first resolved relative to the CSV folder, then the working directory; `INPUT_DIR` is also considered if configured.
- `group`: `GJ` or `GA` (the two age-group codes).

Alternatively, set `METADATA_FILE = None` and point `INPUT_DIR` to a folder of `.set` files. In this mode the subject ID is the filename stem and the group is inferred from the **first two characters of the filename**, so filenames must begin with `GJ` or `GA` for valid group identification.

### Conditions and event codes

The defined semantic order is **`PA`, `PC`, `A`, `C`, `AC`, `U`**. The mapping in `constant_variable.py` is:

| SOA code | PA | PC | A | C | AC | U | Pseudoword |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 33 | 34 | 35 | 36 | 37 | 38 | 39 |
| 2 | 49 | 50 | 51 | 52 | 53 | 54 | 55 |

**Important:** SOA values `1` and `2` are event **codes**, not verified millisecond durations. Verify any association with 150 ms and 300 ms against the experiment log before reporting those durations. The block length defaults to **104 original stimulus markers**; verify this matches the recording and task structure, including practice or incomplete blocks.

## Configuration

Edit `constant_variable.py` before running. The scripts are configured from this file and **do not require command-line arguments**.

```python
METADATA_FILE = PROJECT_DIR / "meta_data.csv"
INPUT_DIR = None
OUTPUT_DIR = PROJECT_DIR / "RSA_results"
COUNTS_FILE = None  # optional trial-count QC CSV

BLOCK_SIZE = 104
WINDOW_SAMPLES = 1
STEP_SAMPLES = 1
SHRINK = 0.15
RANK_TOL = 1e-7
MIN_PER_CELL = 1
INCLUDE_MASTOIDS = False
PARTITION_MODE = "odd_even"  # or "strict_odd_even"
N_PERM = 2000
LIMIT = None
```

The default `WINDOW_SAMPLES = 1` and `STEP_SAMPLES = 1` compute one RDM per sampled EEG time point. Increasing the window averages voltages within each window before estimating distances, and can change the interpretation and time resolution.

An optional `COUNTS_FILE` can be supplied for checking event-filtered trial totals. The CSV is expected to include `file_path` and `n_total` columns.

## Run the primary analysis

From the project folder:

```bash
python main.py
```

Or open `main.py` in PyCharm and select **Run** after configuring `constant_variable.py`.

For a quick trial, set `LIMIT = 1` in the configuration, inspect the QC output, and then restore `LIMIT = None` for the full dataset. With fewer than five successfully processed participants, the main analysis skips its group-level inferential statistics.

### Processing workflow

1. Read participant paths and group labels from the manifest (or discover `.set` files via `INPUT_DIR`).
2. Recover stimulus events and original stimulus-block indices from EEGLAB event metadata.
3. Divide complete original blocks into independent analysis partitions A and B. Odd/even blocks are preferred; the default mode can fall back to another event-count-balanced block split without using EEG values or p-values. `strict_odd_even` disables that fallback.
4. Read EEG epochs with MNE, exclude EOG channels and (by default) mastoids, and check epoch alignment and minimum usable channels.
5. Compute condition means within each partition and SOA, estimate shrinkage-regularized noise precision from within-condition residuals, and calculate time-resolved RDMs.
6. Save subject results and QC information, then generate pooled group figures and, when sample size permits, statistical summaries.

### Representational measures

The analysis computes 6 × 6 RDMs per SOA and time point with:

- **Crossnobis:** a cross-partition, noise-precision-weighted squared distance, normalized by estimated noise rank. Negative estimates are possible and should not automatically be clipped.
- **Cross-validated squared Euclidean distance:** a cross-partition squared-distance estimate normalized by channel count.
- **Correlation distance:** `1 - correlation` between pooled condition patterns; unlike the first two measures, this is **not cross-validated**.

The results also contain the `QP`, `QA`, and `QC` shared-pattern indices, a cross-validated additivity-deviation measure, and theoretical contrasts:

| Contrast | Interpretation in the code |
| --- | --- |
| H1 (`h1`) | PA–PC proximity relative to selected between-relation distances. |
| H1b (`h1_distinct`) | Direct PA–PC distance (discriminability). |
| H2 (`h2`) | A/C/AC distances relative to across-category distances. |
| H3 (`h3`) | A-specific representational organization. |
| H4 (`h4`) | Signed, whitened cross-partition interaction for `AC - A - C + U`. |

The **primary semantic pairwise results** focus on the ten pairs among `PA`, `PC`, `A`, `C`, and `AC`; `U` is excluded from these primary comparisons but retained in the saved full six-condition RDMs and in the H4 interaction definition.

## Primary outputs

By default, outputs are written to `RSA_results/`:

```text
RSA_results/
├── quality_control.csv
├── analysis_config.json
├── <subject>_rsa.npz           # one per successful participant
├── group_data.npz
├── related_pair_by_time.npz
├── related_pair_timepoint_stats.csv  # if enough participants
├── theoretical_contrasts_timepoint_stats.csv  # if enough participants
├── h1_AGE_by_SOA_timeseries.png
├── h1_distinct_AGE_by_SOA_timeseries.png
├── h2_AGE_by_SOA_timeseries.png
├── h3_AGE_by_SOA_timeseries.png
├── h4_AGE_by_SOA_timeseries.png
├── semantic_related_pair_by_time_heatmap.png
├── soa_related_pair_difference_heatmap.png
└── ... other related-pair, SOA, age-group and RDM figures
```

The `quality_control.csv` includes processing status, recognized stimulus counts, partition assignments, and (when successful) channel count, smallest condition-cell count and median covariance rank. Participant archives retain RDMs, contrasts, timestamps, partition information, channels and condition counts; `group_data.npz` gathers key arrays across successful participants.

Some plot and age-comparison files are only created if both age groups are represented. Inferential output depends on minimum sample sizes and data validity.

## Optional H2/H4 sensitivity analysis

**Run `main.py` first** to create the per-subject `*_rsa.npz` files. Then run:

```bash
python sensitivity_h2_h4.py
```

The optional analysis uses `SENS_*` settings in `constant_variable.py` and, by default, reads from `RSA_results/` and writes to `RSA_H2_H4_sensitivity/`.

It:

- Compares original, 3-sample and 5-sample H2/H4 time series. Smoothing is applied to **already computed contrasts**, not raw EEG signals or input RDMs.
- Performs exploratory max-|t| permutation testing, jointly correcting across H2/H4, both smoothing widths and time points **within each contrast family** (semantic mean, SOA, age difference, and age × SOA).
- Estimates an exploratory lag maximizing correlation between group-average H2 curves for a configured SOA and time interval, with subject-level bootstrap uncertainty, if each age group has at least three participants.

Selected outputs include:

```text
RSA_H2_H4_sensitivity/
├── analysis_settings.json
├── H2_H4_original_3pt_5pt_curves.png
├── H2_H4_smoothed_maxT_stats.csv
├── H2_SOA1_lag_correlation_grid.csv
├── H2_SOA1_bootstrap_lags_ms.csv
├── H2_SOA1_shift_summary.json
├── H2_GJ_GA_SOA1_lag_correlation.png
└── H2_lag_bootstrap_distribution.png
```

The lag analysis is **exploratory**. A best-matching temporal shift is not an estimate of neural onset, proof of compensatory processing, or a formal significance test of latency. Searching many shifts and selecting the maximum can inflate the observed correlation.

## Statistical and interpretation cautions

- The crossnobis implementation uses a **plug-in shrinkage precision estimator**; it is not guaranteed to be finite-sample unbiased.
- The primary related-pair permutation tests correct across the **10 related pairs and time points within each family**. Semantic, SOA and age are distinct testing families; interpretation across families requires an additional multiplicity strategy or a prespecified hierarchy.
- The theoretical contrast tests use their own defined correction families. Their adjusted p-values should not be taken as globally corrected across every result in the repository.
- The sensitivity analysis jointly corrects H2/H4 and smoothing choices within each statistical contrast, **not across its four contrast families**.
- Descriptive mean/SEM plots do not, by themselves, establish statistical significance.
- The pipeline does **not** run a new EEG baseline correction, rereferencing, or ICA. Input `.set` files should already reflect the intended preprocessing decisions.
- The full-epoch RDM summaries average time-resolved distance estimates; they are **not** distances computed from one grand-average voltage pattern.
- Verify event mappings, epoch alignment, recording reference, and group labels before drawing substantive conclusions.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `ModuleNotFoundError` for a project module | Rename uploaded `*(2).py` files to the expected import names and run from the project directory. |
| Missing `.set` files | Check paths in `meta_data.csv`, especially paths relative to the CSV file, and optional `INPUT_DIR`. |
| No complete two-partition split | Inspect event markers, 104-stimulus block boundaries, missing conditions, and `MIN_PER_CELL`. |
| Epoch count mismatch | Confirm that `.set` event records align one-to-one with EEG epochs as this parser requires. |
| Too few residual trials / invalid covariance | Inspect sparse conditions, retained trials, and preprocessing artifacts. |
| Empty or incomplete group statistics | Check `quality_control.csv`, number of successful subjects, and whether both `GJ` and `GA` are present. |
| SOA duration interpretation is uncertain | Verify code-to-millisecond mapping against the original experiment log. |

## Reproducibility

Keep `meta_data.csv`, preprocessing details, experiment/event logs, the chosen configuration, and the generated `analysis_config.json`/`analysis_settings.json` with the analysis results. The random seeds and permutation counts are configured in `constant_variable.py`; output depends on the installed scientific Python libraries and the underlying input datasets.
