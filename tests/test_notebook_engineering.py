from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "MINING_5_Final.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code() -> str:
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def _loader_namespace() -> dict:
    """Load the notebook's dependency-free loader without executing the data-load call."""
    cells = _notebook()["cells"]
    namespace: dict = {}
    exec("".join(cells[1].get("source", [])), namespace)
    loader_source = "".join(cells[3].get("source", []))
    exec(loader_source.split("df = load_dataset()", 1)[0], namespace)
    return namespace


def test_all_notebook_code_cells_parse() -> None:
    for number, cell in enumerate(_notebook()["cells"], start=1):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell-{number}")


def test_notebook_has_no_stale_outputs_or_execution_counts() -> None:
    for cell in _notebook()["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == []
            assert cell.get("execution_count") is None


def test_data_load_is_guarded_and_no_local_absolute_path_is_hardcoded() -> None:
    code = _code()
    assert "CRIME_DATA_PATH" in code
    assert "FileNotFoundError" in code
    assert "pd.read_csv(path)" in code
    assert "/content/" not in code
    assert "C:\\" not in code


def test_model_features_exclude_targets_and_fit_preprocessor_on_train_only() -> None:
    code = _code()
    assert 'FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES' in code
    assert 'preprocessor.fit_transform(train_df[FEATURE_COLUMNS])' in code
    assert 'preprocessor.transform(X' not in code
    assert 'preprocessor.transform(validation_df[FEATURE_COLUMNS])' in code
    assert 'preprocessor.transform(test_df[FEATURE_COLUMNS])' in code
    assert '"Category"' not in code.split("FEATURE_COLUMNS", 1)[1].split("sample_size", 1)[0]


def test_repository_has_dependency_and_ignore_contracts() -> None:
    assert (ROOT / "requirements.txt").is_file()
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "train.csv" in ignore
    assert "artifacts/" in ignore


def test_loader_rejects_missing_explicit_path_instead_of_falling_back(tmp_path: Path) -> None:
    loader = _loader_namespace()
    missing = tmp_path / "missing.csv"
    try:
        loader["resolve_data_path"](missing)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("A missing explicit data path must fail loudly.")


def test_loader_drops_invalid_rows_and_parses_mixed_timezone_dates(tmp_path: Path) -> None:
    import pandas as pd

    loader = _loader_namespace()
    rows = [
        {
            "Dates": "2015-05-13 23:53:00",
            "Category": "LARCENY/THEFT",
            "Descript": "THEFT",
            "DayOfWeek": "Wednesday",
            "PdDistrict": "NORTHERN",
            "Resolution": "NONE",
            "Address": "1 MARKET ST",
            "X": "-122.4",
            "Y": "37.8",
        },
        {
            "Dates": "2015-05-14T00:53:00+01:00",
            "Category": " ",
            "Descript": "THEFT",
            "DayOfWeek": "Thursday",
            "PdDistrict": "NORTHERN",
            "Resolution": "NONE",
            "Address": "2 MARKET ST",
            "X": "-122.4",
            "Y": "37.8",
        },
    ]
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    frame = loader["load_dataset"](path)
    assert len(frame) == 1
    assert str(frame["Dates"].dtype).startswith("datetime64[ns, UTC]")
    assert frame["Hour"].tolist() == [23]
