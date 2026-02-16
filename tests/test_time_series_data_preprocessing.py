import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class TestTimeSeriesPreprocessingScript(unittest.TestCase):
    def test_train_only_feature_selection_avoids_test_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            owid_dir = tmp_path / "owid" / "climate_data"
            owid_dir.mkdir(parents=True, exist_ok=True)

            table_csv = owid_dir / "100.csv"
            table_meta = owid_dir / "100.meta.json"
            policy_csv = tmp_path / "group_kfold_assignments.csv"
            output_dir = tmp_path / "output"

            pd.DataFrame(
                [
                    {
                        "country": "Trainland",
                        "year": 2000,
                        "feat_common": 1.0,
                        "feat_test_only": None,
                    },
                    {
                        "country": "Trainland",
                        "year": 2001,
                        "feat_common": 2.0,
                        "feat_test_only": None,
                    },
                    {
                        "country": "Testland",
                        "year": 2000,
                        "feat_common": 3.0,
                        "feat_test_only": 10.0,
                    },
                    {
                        "country": "Testland",
                        "year": 2001,
                        "feat_common": 4.0,
                        "feat_test_only": 11.0,
                    },
                ]
            ).to_csv(table_csv, index=False)

            with table_meta.open("w", encoding="utf-8") as handle:
                json.dump({"title": "Synthetic table"}, handle)

            pd.DataFrame(
                [
                    {"Geography": "Trainland", "fold": 1},
                    {"Geography": "Testland", "fold": 0},
                ]
            ).to_csv(policy_csv, index=False)

            repo_root = Path(__file__).resolve().parents[1]
            script_path = repo_root / "scripts" / "1_time_series_data_preprocessing.py"

            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--owid-dir",
                    str(tmp_path / "owid"),
                    "--policy-input",
                    str(policy_csv),
                    "--output-dir",
                    str(output_dir),
                    "--start-year",
                    "2000",
                    "--end-year",
                    "2001",
                    "--fold",
                    "0",
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)

            train_country_file = (
                output_dir / "kfold" / "fold_0" / "train" / "Trainland.csv"
            )
            test_country_file = (
                output_dir / "kfold" / "fold_0" / "test" / "Testland.csv"
            )

            self.assertTrue(train_country_file.exists())
            self.assertTrue(test_country_file.exists())

            test_df = pd.read_csv(test_country_file)

            self.assertIn("t000__feat_common", test_df.columns)
            self.assertNotIn("t000__feat_test_only", test_df.columns)


if __name__ == "__main__":
    unittest.main()
