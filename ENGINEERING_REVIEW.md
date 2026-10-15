# Engineering review — Crime_Data_Mining

**Repository:** `Abdulrahman181/Crime_Data_Mining`  
**Audit date:** 2026-10-14  
**Scope:** all repository Python/notebook/document/configuration artifacts; no other repository was changed.

## Executive summary

The repository contained one stale executed notebook, a short README, and a PDF report. It did not contain the referenced `train.csv`, `requirements.txt`, tests, `.gitignore`, or Streamlit source. The notebook could not be responsibly run end-to-end in this Sandbox because the real dataset is absent and `scikit-learn-extra`/Jupyter are not installed. No metrics or replacement data were fabricated.

The notebook was rebuilt as a clean 17-cell workflow and all saved outputs/execution counts were removed. It now has explicit data/schema validation, deterministic sampling and seeds, train/validation/test separation, train-only preprocessing, optional dependency errors, bounded clustering, metrics appropriate for imbalanced pseudo-labels, and a clear limitations section.

## Findings in the original repository

- **Reproducibility:** `pd.read_csv("train.csv")` depended on the current working directory; no data contract or dependency file existed.
- **Stale artifacts:** the notebook had 39 code cells, 30 cells with saved outputs, old execution counts, Colab metadata, and plots/metrics that could not be reproduced from the repository.
- **Data leakage:** scaling and coordinate outlier thresholds were fitted before any modeling split. K-Medoids labels were created from the full sampled dataset and only then split for the supervised experiment. This makes the classifier evaluation optimistic and transductive. The original split was also effectively 60% train / 32% validation / 8% test, despite comments describing a 60/20/20 split.
- **Feature semantics:** the original workflow used ordinal label encoding for a categorical district in Euclidean clustering. The repaired workflow uses a train-fitted one-hot encoder and keeps `Category`, `Descript`, `Resolution`, and `Address` out of model inputs.
- **Stability:** several models/steps lacked explicit seeds; the tree had no `random_state`; clustering was fit on all sampled rows; and full-data preprocessing could change with future rows.
- **Dependency/API risks:** `sklearn_extra.cluster.KMedoids` was imported unconditionally even though the dependency was not declared. `AgglomerativeClustering(metric=...)` requires a sufficiently recent scikit-learn version. The repository had no application/API implementation despite the historical README/report mentioning Streamlit.
- **Security:** no credentials, tokens, passwords, unsafe deserialization, shell execution, or API endpoints were found. The only network-facing references are the Kaggle and historical Streamlit links in documentation.

## Repairs made

- Replaced `MINING_5_Final.ipynb` with a clean, unexecuted notebook:
  - resolves `CRIME_DATA_PATH`, root `train.csv`, or `data/train.csv`;
  - raises actionable errors for missing files, schema mismatch, unreadable data, and zero valid rows;
  - removes exact duplicates and invalid date/coordinate rows without inventing values;
  - uses a deterministic maximum 5,000-row sample (`random_state=42`) to bound clustering cost;
  - splits 60/20/20 before fitting the `ColumnTransformer`;
  - fits scaling/one-hot encoding on training rows only;
  - selects K from training-only silhouette scores;
  - evaluates clustering with silhouette, Calinski-Harabasz, and Davies-Bouldin;
  - treats classifier targets honestly as K-Medoids pseudo-labels and reports accuracy, balanced accuracy, macro F1, and a test report;
  - uses explicit seeds and `n_jobs=-1` only for the Random Forest;
  - writes derived plots only to ignored `artifacts/`.
- Rewrote `README.md` with setup, data contract, commands, methodology, leakage safeguards, limitations, and an explicit note that no Streamlit source exists here.
- Added `requirements.txt` with bounded runtime, notebook, testing, and Ruff dependencies.
- Added `.gitignore` for raw data, artifacts, environments, caches, and notebook checkpoints.
- Added `pyproject.toml` Ruff/pytest configuration.
- Added `data/README.md` documenting the non-committed input contract.
- Added `tests/test_notebook_engineering.py` covering AST parsing, cleared outputs, data guards, local-path hygiene, feature/split safeguards, and dependency/ignore contracts.

## Validation evidence

The following checks passed after the repairs:

- `ruff check .` — **passed**.
- `pytest -q` — **5 passed**.
- AST parsing of all 9 notebook code cells — **passed**.
- Notebook output audit — **0 saved outputs; every code cell has `execution_count: null`**.
- Missing-data guard execution — raised the documented `FileNotFoundError` and listed the searched paths.
- Repository scan — no hard-coded `/content/`, `/Users/`, Windows local paths, or secret-like credentials found.

## Not run / limitations

- End-to-end modeling was **not run**: no real `train.csv` exists in the repository or Sandbox. The notebook therefore makes no performance claim.
- `scikit-learn-extra` and Jupyter are not installed in the current environment; the dependency is declared, and the notebook gives an explicit install error at the K-Medoids cell.
- The historical PDF report was retained as supplied and is not treated as current experimental evidence. Its old metrics/claims were not copied into the repaired README or notebook.
- No Streamlit application was invented because no source or deployable app exists in this repository.

## Second Senior Review — 2026-10-15

### Scope and findings

The second review re-checked every tracked file, all 17 notebook cells, the prior audit commit (`2a2728f`), Git history, dependency declarations, and the existing tests. The first repair was materially sound, but the following residual issues were found:

- A missing explicit `CRIME_DATA_PATH` or `load_dataset` path could silently fall back to a different local CSV, which is unsafe for reproducibility.
- `pd.to_datetime` could leave mixed-timezone input as an object dtype and then fail at `.dt` feature extraction; blank model-critical categorical values were also accepted.
- Very small datasets reached a less actionable split/clustering failure, and a training set containing fewer than two distinct feature rows could make silhouette/K-Medoids invalid.
- The confusion-matrix cell omitted labels that appeared only in predictions, so counts could be silently dropped from the displayed matrix.
- The dependency range allowed NumPy 2 and newer scikit-learn versions that are not a safe compatibility target for the unmaintained `scikit-learn-extra==0.3.0` package.
- Plot figures were left open after saving, which is avoidable state/memory accumulation during repeated notebook execution.

No API/UI implementation, model serialization, credentials, unsafe deserialization, or fabricated data/metrics were found. The historical PDF remains non-executable legacy evidence; the repository still has no Streamlit source.

### Repairs made

- Made an explicit data path authoritative: a missing configured path now raises `FileNotFoundError` instead of falling back; documented fallback search remains available only when no path is configured.
- Parsed dates with `utc=True`, rejected blank `Category`, `DayOfWeek`, and `PdDistrict` values, and retained the existing no-imputation/finite-coordinate safeguards.
- Added a minimum-five-row guard for the 60/20/20 split and a distinct-feature-row guard plus bounded K selection for clustering.
- Included the union of true and predicted labels in the test confusion matrix and close saved Matplotlib figures after display.
- Constrained requirements to `numpy<2`, `scikit-learn>=1.3,<1.6`, and `scikit-learn-extra==0.3.0` to make the declared legacy K-Medoids stack explicit.
- Added executable loader tests for strict path precedence, mixed-timezone parsing, and blank-row rejection; updated README behavior and limitations.

### Second-review validation

- `ruff check .` — passed.
- `pytest -q` — passed (7 tests).
- `python -m compileall -q .` — passed for Python sources/tests.
- Notebook JSON/AST/output audit — passed: all code cells parse, have empty outputs, and have null execution counts.
- Full end-to-end modeling — **not run**: no real `train.csv` is present and `scikit-learn-extra`/`nbformat` are not installed in the current sandbox. A Python 3.12 isolated install also attempted a source build for `scikit-learn-extra==0.3.0` and lacked Python headers; Python 3.9–3.11 is documented as the supported setup. No performance metrics are claimed.

## Local commit

A local-only commit is created after this second review. No push or remote mutation is performed.
