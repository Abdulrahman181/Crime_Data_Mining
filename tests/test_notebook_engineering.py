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
