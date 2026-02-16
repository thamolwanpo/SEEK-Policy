import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class TestSplitTestSetScript(unittest.TestCase):
    def test_grouped_kfold_train_test_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_csv = tmp_path / "all_data_en.csv"
            region_csv = tmp_path / "region.csv"
            output_dir = tmp_path / "out"

            policy_df = pd.DataFrame(
                [
                    {
                        "Document ID": "doc_f1_v1",
                        "Family ID": "FAMILY_1",
                        "Geography ISO": "THA",
                        "Last event in timeline": "2008-01-01",
                    },
                    {
                        "Document ID": "doc_f1_v2",
                        "Family ID": "FAMILY_1",
                        "Geography ISO": "THA",
                        "Last event in timeline": "2018-01-01",
                    },
                    {
                        "Document ID": "doc_f2",
                        "Family ID": "FAMILY_2",
                        "Geography ISO": "USA",
                        "Last event in timeline": "2009-01-01",
                    },
                    {
                        "Document ID": "doc_f3",
                        "Family ID": "FAMILY_3",
                        "Geography ISO": "JPN",
                        "Last event in timeline": "2010-01-01",
                    },
                    {
                        "Document ID": "doc_f4",
                        "Family ID": "FAMILY_4",
                        "Geography ISO": "KOR",
                        "Last event in timeline": "2011-01-01",
                    },
                    {
                        "Document ID": "doc_f5",
                        "Family ID": "FAMILY_5",
                        "Geography ISO": "FRA",
                        "Last event in timeline": "2012-01-01",
                    },
                    {
                        "Document ID": "doc_f6",
                        "Family ID": "FAMILY_6",
                        "Geography ISO": "BRA",
                        "Last event in timeline": "2013-01-01",
                    },
                ]
            )
            policy_df.to_csv(input_csv, index=False)

            region_df = pd.DataFrame(
                [
                    {
                        "alpha-3": "THA",
                        "region": "Asia",
                        "sub-region": "SE Asia",
                        "intermediate-region": "",
                    },
                    {
                        "alpha-3": "USA",
                        "region": "Americas",
                        "sub-region": "Northern America",
                        "intermediate-region": "",
                    },
                    {
                        "alpha-3": "JPN",
                        "region": "Asia",
                        "sub-region": "East Asia",
                        "intermediate-region": "",
                    },
                    {
                        "alpha-3": "KOR",
                        "region": "Asia",
                        "sub-region": "East Asia",
                        "intermediate-region": "",
                    },
                    {
                        "alpha-3": "FRA",
                        "region": "Europe",
                        "sub-region": "Western Europe",
                        "intermediate-region": "",
                    },
                    {
                        "alpha-3": "BRA",
                        "region": "Americas",
                        "sub-region": "South America",
                        "intermediate-region": "",
                    },
                ]
            )
            region_df.to_csv(region_csv, index=False)

            repo_root = Path(__file__).resolve().parents[1]
            script_path = repo_root / "scripts" / "0_split_test_set.py"

            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--input",
                    str(input_csv),
                    "--region",
                    str(region_csv),
                    "--output-dir",
                    str(output_dir),
                    "--n-folds",
                    "3",
                    "--seed",
                    "42",
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)

            fold_df = pd.read_csv(output_dir / "group_kfold_assignments.csv")
            kfold_dir = output_dir / "kfold"

            # Reviewer requirement 1: grouped folds keep each family in exactly one fold.
            folds_per_family = fold_df.groupby("Family ID")["fold"].nunique()
            self.assertTrue((folds_per_family == 1).all())
            self.assertTrue(set(fold_df["fold"].unique()).issubset({0, 1, 2}))

            # Explicit reviewer case: 2008 and 2018 versions of the same law family
            # must remain in one fold (never train/test split).
            family_1_fold_count = fold_df.loc[
                fold_df["Family ID"] == "FAMILY_1", "fold"
            ].nunique()
            self.assertEqual(family_1_fold_count, 1)

            # Reviewer requirement 2: each fold has train-test files with no group leakage.
            for fold_id in [0, 1, 2]:
                train_path = kfold_dir / f"fold_{fold_id}_train.csv"
                test_path = kfold_dir / f"fold_{fold_id}_test.csv"
                self.assertTrue(train_path.exists())
                self.assertTrue(test_path.exists())

                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
                self.assertGreater(len(train_df), 0)
                self.assertGreater(len(test_df), 0)

                train_families = set(train_df["Family ID"].unique())
                test_families = set(test_df["Family ID"].unique())
                self.assertEqual(train_families.intersection(test_families), set())


if __name__ == "__main__":
    unittest.main()
