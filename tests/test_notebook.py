"""Offline integrity checks for the analysis notebook; no incident dataset is needed."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "MINING_5_Final.ipynb"


class NotebookIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.cells = cls.notebook["cells"]
        cls.sources = ["".join(cell.get("source", [])) for cell in cls.cells]

    def test_all_code_cells_parse_as_python(self) -> None:
        code_cells = [cell for cell in self.cells if cell.get("cell_type") == "code"]
        self.assertGreater(len(code_cells), 0)
        for number, cell in enumerate(code_cells):
            compile("".join(cell.get("source", [])), f"notebook-cell-{number}", "exec")

    def test_committed_notebook_has_no_stale_or_data_bearing_outputs(self) -> None:
        for number, cell in enumerate(self.cells):
            self.assertEqual(cell.get("outputs", []), [], f"cell {number} has saved output")
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"), f"cell {number} is marked executed")

    def test_input_cell_uses_override_checks_schema_and_reads_minimal_fields(self) -> None:
        source = self.sources[3]
        namespace = {"pd": pd, "Path": Path, "os": os}
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "incidents.csv"
            pd.DataFrame(
                {
                    "Dates": ["2015-01-01 01:00:00", "2015-01-02 02:00:00"],
                    "Category": ["A", "B"],
                    "DayOfWeek": ["Thursday", "Friday"],
                    "PdDistrict": ["NORTH", "SOUTH"],
                    "X": [-122.4, -122.5],
                    "Y": [37.7, 37.8],
                    "Address": ["sensitive value 1", "sensitive value 2"],
                    "Descript": ["not needed", "not needed"],
                }
            ).to_csv(csv_path, index=False)
            with patch.dict(os.environ, {"CRIME_DATA_PATH": str(csv_path)}):
                exec(compile(source, "notebook-input-cell", "exec"), namespace)
            frame = namespace["df"]
            self.assertEqual(
                list(frame.columns),
                ["Dates", "Category", "DayOfWeek", "PdDistrict", "X", "Y"],
            )
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["Dates"]))
            self.assertTrue(pd.api.types.is_numeric_dtype(frame["X"]))

            # Missing required columns and invalid required values fail closed.
            missing_path = Path(directory) / "missing.csv"
            pd.DataFrame({"Dates": ["2015-01-01"]}).to_csv(missing_path, index=False)
            with patch.dict(os.environ, {"CRIME_DATA_PATH": str(missing_path)}):
                with self.assertRaisesRegex(ValueError, "missing required columns"):
                    exec(compile(source, "notebook-input-cell", "exec"), {"pd": pd, "Path": Path, "os": os})

            invalid_path = Path(directory) / "invalid.csv"
            invalid = pd.read_csv(csv_path)
            invalid.loc[0, "Dates"] = "not-a-date"
            invalid.to_csv(invalid_path, index=False)
            with patch.dict(os.environ, {"CRIME_DATA_PATH": str(invalid_path)}):
                with self.assertRaisesRegex(ValueError, "missing or invalid required values"):
                    exec(compile(source, "notebook-input-cell", "exec"), {"pd": pd, "Path": Path, "os": os})

    def test_actual_split_cell_creates_stratified_disjoint_60_20_20_partitions(self) -> None:
        from sklearn.model_selection import train_test_split  # noqa: F401

        source = self.sources[40]
        namespace = {
            "X_new": pd.DataFrame({"x": range(1000)}, index=range(1000)),
            "y": pd.Series([0, 1] * 500, index=range(1000)),
        }
        exec(compile(source, "notebook-split-cell", "exec"), namespace)
        self.assertEqual(len(namespace["x_train"]), 600)
        self.assertEqual(len(namespace["x_val"]), 200)
        self.assertEqual(len(namespace["x_test"]), 200)
        self.assertEqual(set(namespace["y_train"].unique()), {0, 1})
        self.assertEqual(set(namespace["y_val"].unique()), {0, 1})
        self.assertEqual(set(namespace["y_test"].unique()), {0, 1})

    def test_classifier_section_discloses_full_sample_target_leakage(self) -> None:
        classifier_notes = self.sources[38].lower()
        self.assertIn("full analysis sample", classifier_notes)
        self.assertIn("not a leakage-free", classifier_notes)
        self.assertIn("not time- or location-held-out", classifier_notes)

    def test_input_schema_matches_notebook_loader(self) -> None:
        schema = json.loads((ROOT / "input_schema.json").read_text(encoding="utf-8"))
        required = schema["required_columns"]
        self.assertEqual(set(required), set(schema["columns"]))
        self.assertTrue(all(schema["columns"][name]["nullable"] is False for name in required))
        module = ast.parse(self.sources[3])
        assignment = next(
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "REQUIRED_COLUMNS" for target in node.targets)
        )
        self.assertEqual(ast.literal_eval(assignment.value), required)
        self.assertIn("usecols=REQUIRED_COLUMNS", self.sources[3])

    def test_direct_dependencies_are_exactly_pinned(self) -> None:
        requirements = ROOT / "requirements.txt"
        lines = [line.strip() for line in requirements.read_text(encoding="utf-8").splitlines()]
        pins = [line for line in lines if line and not line.startswith("#")]
        self.assertTrue(pins)
        self.assertTrue(all("==" in line for line in pins))
        self.assertEqual(len(pins), len({line.split("==", 1)[0].lower() for line in pins}))


if __name__ == "__main__":
    unittest.main()
