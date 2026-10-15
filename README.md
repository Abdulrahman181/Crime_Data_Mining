# Crime Data Mining

A reproducible analysis of the San Francisco crime dataset using descriptive analysis, K-Medoids and hierarchical clustering, and supervised models that reproduce K-Medoids pseudo-labels. The maintained executable artifact is [`MINING_5_Final.ipynb`](MINING_5_Final.ipynb).

## Data

The repository intentionally does **not** include the large source dataset or generated artifacts. Obtain the SF Crime dataset from the [Kaggle SF Crime Rate Prediction page](https://www.kaggle.com/code/junheo/sf-crime-rate-prediction), then save the CSV as `train.csv` at the repository root or under `data/train.csv`. Alternatively set `CRIME_DATA_PATH` to an explicit local CSV path.

The loader requires these columns:

`Dates`, `Category`, `Descript`, `DayOfWeek`, `PdDistrict`, `Resolution`, `Address`, `X`, `Y`.

No data is downloaded, synthesized, or fabricated by the notebook. If `CRIME_DATA_PATH` (or the optional `load_dataset` argument) is set but does not point to a file, the first loading cell stops rather than silently falling back to another path. If no path is configured, the notebook searches the current working directory and its `data/` subdirectory, then raises an actionable `FileNotFoundError`.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Use Python 3.9–3.11 for the complete workflow. `scikit-learn-extra==0.3.0` is an old package and may require a local build on Python 3.12+; the notebook reports a clear install error if K-Medoids is unavailable.

For notebook execution:

```bash
jupyter notebook MINING_5_Final.ipynb
# or: CRIME_DATA_PATH=/absolute/path/train.csv jupyter nbconvert --to notebook --execute MINING_5_Final.ipynb --output executed.ipynb
```

Generated plots go to `artifacts/` by default. Override that directory with `CRIME_ARTIFACT_DIR`.

## Methodology and safeguards

- Exact duplicates and invalid dates/coordinates are removed; no synthetic or silent imputation is performed.
- Modeling uses only date-derived fields, coordinates, and one-hot police-district features. `Category`, `Descript`, `Resolution`, and `Address` are excluded from model inputs.
- A deterministic sample (maximum 5,000 rows, seed `42`) bounds clustering cost.
- At least five valid rows are required for the documented 60/20/20 split; at least two distinct training feature rows are required for clustering.
- The train/validation/test split is 60/20/20. The feature preprocessor is fit on training rows only, then applied to validation and test rows.
- K is selected using training-only silhouette scores. Clustering diagnostics are reported on training rows.
- Decision Tree and Random Forest results predict K-Medoids cluster pseudo-labels; they are **not** crime-category prediction, causal inference, or forecasts of future incidents. Report accuracy alongside balanced accuracy and macro F1 because cluster sizes may be uneven.

## Validation and development checks

```bash
ruff check .
pytest -q
```

The tests cover notebook syntax/state, loader validation/path precedence, required data guards, and the documented split/feature constraints. A complete model run requires the real dataset and installed optional scientific dependencies; no metrics are claimed in this repository without those inputs.

## Repository scope

The historical project report and README referenced a Streamlit dashboard, but no Streamlit source exists in this repository. This audit does not invent or claim a dashboard implementation; the notebook is the maintained analysis entry point.
