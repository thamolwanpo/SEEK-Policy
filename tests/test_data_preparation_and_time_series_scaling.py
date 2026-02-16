import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class TestDataPreparationAndScalingScript(unittest.TestCase):
    def test_generates_multi_window_ablation_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            policy_csv = tmp_path / "group_kfold_assignments.csv"
            time_series_root = tmp_path / "time_series" / "kfold" / "fold_0"
            output_dir = tmp_path / "model_input"

            (time_series_root / "train").mkdir(parents=True, exist_ok=True)
            (time_series_root / "test").mkdir(parents=True, exist_ok=True)

            policy_df = pd.DataFrame(
                [
                    {
                        "Document ID": "train_doc_1",
                        "Family Summary": "Train policy one",
                        "Sector": "Energy;Transport",
                        "Geography": "Trainland",
                        "Last event in timeline": "2004-01-01",
                        "fold": 1,
                    },
                    {
                        "Document ID": "train_doc_2",
                        "Family Summary": "Train policy two",
                        "Sector": "Health",
                        "Geography": "Trainland2",
                        "Last event in timeline": "2004-01-01",
                        "fold": 1,
                    },
                    {
                        "Document ID": "test_doc_1",
                        "Family Summary": "Test policy one",
                        "Sector": "Energy",
                        "Geography": "Testland",
                        "Last event in timeline": "2004-01-01",
                        "fold": 0,
                    },
                ]
            )
            policy_df.to_csv(policy_csv, index=False)

            trainland = pd.DataFrame(
                {
                    "country": ["Trainland"] * 5,
                    "year": [2000, 2001, 2002, 2003, 2004],
                    "feature_a": [1, 2, 3, 4, 5],
                    "feature_b": [10, 10, 11, 12, 13],
                }
            )
            trainland2 = pd.DataFrame(
                {
                    "country": ["Trainland2"] * 5,
                    "year": [2000, 2001, 2002, 2003, 2004],
                    "feature_a": [2, 3, 4, 5, 6],
                    "feature_b": [9, 10, 10, 11, 12],
                }
            )
            testland = pd.DataFrame(
                {
                    "country": ["Testland"] * 5,
                    "year": [2000, 2001, 2002, 2003, 2004],
                    "feature_a": [5, 6, 7, 8, 9],
                    "feature_b": [11, 12, 13, 14, 15],
                }
            )

            trainland.to_csv(time_series_root / "train" / "Trainland.csv", index=False)
            trainland2.to_csv(
                time_series_root / "train" / "Trainland2.csv", index=False
            )
            testland.to_csv(time_series_root / "test" / "Testland.csv", index=False)

            repo_root = Path(__file__).resolve().parents[1]
            script_path = (
                repo_root / "scripts" / "2_data_preparation_and_time_series_scaling.py"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--policy-input",
                    str(policy_csv),
                    "--time-series-dir",
                    str(tmp_path / "time_series" / "kfold"),
                    "--output-dir",
                    str(output_dir),
                    "--windows",
                    "1,2,5,10",
                    "--fold",
                    "0",
                    "--negative-samples",
                    "1",
                    "--seed",
                    "123",
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)

            fold_out = output_dir / "fold_0"
            self.assertTrue((fold_out / "scaled_train_time_series.csv").exists())
            self.assertTrue((fold_out / "scaled_test_time_series.csv").exists())
            self.assertTrue((fold_out / "scaler.pkl").exists())

            for window in [1, 2, 5, 10]:
                window_dir = fold_out / f"window_{window}"
                train_jsonl = window_dir / "train.jsonl"
                test_json = window_dir / "test.json"

                self.assertTrue(train_jsonl.exists())
                self.assertTrue(test_json.exists())

                train_rows = [
                    json.loads(line)
                    for line in train_jsonl.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertGreater(len(train_rows), 0)

                first_train = train_rows[0]
                self.assertEqual(len(first_train["positive_time_series"]), window)
                self.assertEqual(len(first_train["negative_time_series"]), window)

                test_rows = json.loads(test_json.read_text(encoding="utf-8"))
                self.assertGreater(len(test_rows), 0)
                self.assertEqual(len(test_rows[0]["positive_time_series"]), window)

            summary = json.loads(
                (output_dir / "preparation_summary.json").read_text(encoding="utf-8")
            )
            self.assertIn("folds", summary)
            self.assertEqual(len(summary["folds"]), 1)


if __name__ == "__main__":
    unittest.main()
