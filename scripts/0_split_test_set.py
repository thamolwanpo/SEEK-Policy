import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/csv/all_data_en.csv")
DEFAULT_REGION = Path("data/csv/region.csv")
DEFAULT_OUTPUT_DIR = Path("data/csv")
DEFAULT_KFOLD_DIRNAME = "kfold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create leakage-safe grouped K-fold train/test splits for POLiMATCH, "
            "keeping all versions of a law in the same fold."
        )
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="Path to all_data_en.csv"
    )
    parser.add_argument(
        "--region", type=Path, default=DEFAULT_REGION, help="Path to region.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSV files",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of grouped cross-validation folds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic fold assignment",
    )
    parser.add_argument(
        "--kfold-dir-name",
        type=str,
        default=DEFAULT_KFOLD_DIRNAME,
        help="Subdirectory name under output-dir for per-fold train/test files",
    )
    return parser.parse_args()


def first_valid_series(
    df: pd.DataFrame, candidates: list[str], fallback_name: str
) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            series = df[column].fillna("").astype(str).str.strip()
            if series.ne("").any():
                return series
    return pd.Series([fallback_name] * len(df), index=df.index, dtype="object")


def build_group_key(df: pd.DataFrame) -> pd.Series:
    key = first_valid_series(
        df,
        ["Family ID", "Internal Family ID", "Family Title", "Document ID", "doc_id"],
        "UNKNOWN_FAMILY",
    )
    return key.replace("", "UNKNOWN_FAMILY")


def assign_group_folds(groups: pd.Index, n_folds: int, seed: int) -> pd.Series:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if n_folds > len(groups):
        raise ValueError(
            f"n_folds ({n_folds}) cannot exceed number of unique groups ({len(groups)})."
        )

    rng = np.random.default_rng(seed)
    shuffled = np.array(groups)
    rng.shuffle(shuffled)

    fold_ids = np.arange(len(shuffled)) % n_folds
    return pd.Series(fold_ids, index=shuffled)


def main() -> None:
    args = parse_args()

    policy_df = pd.read_csv(args.input)
    region_df = pd.read_csv(args.region)

    region_cols = ["alpha-3", "region", "sub-region", "intermediate-region"]
    available_region_cols = [col for col in region_cols if col in region_df.columns]
    if "alpha-3" in available_region_cols and "Geography ISO" in policy_df.columns:
        policy_df = policy_df.merge(
            region_df[available_region_cols],
            left_on="Geography ISO",
            right_on="alpha-3",
            how="left",
        )

    policy_df = policy_df.copy()
    policy_df["__group_key"] = build_group_key(policy_df)

    unique_groups = pd.Index(policy_df["__group_key"].drop_duplicates())
    fold_assignments = assign_group_folds(unique_groups, args.n_folds, args.seed)

    fold_df = policy_df[["__group_key"]].copy()
    fold_df["fold"] = fold_df["__group_key"].map(fold_assignments).astype(int)

    base_columns = [col for col in policy_df.columns if not col.startswith("__")]
    full_with_folds = policy_df[base_columns].copy()
    full_with_folds["fold"] = fold_df["fold"].values

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = args.output_dir / "group_kfold_assignments.csv"
    full_with_folds.to_csv(assignment_path, index=False)

    kfold_dir = args.output_dir / args.kfold_dir_name
    kfold_dir.mkdir(parents=True, exist_ok=True)

    print("=== Grouped K-fold split completed ===")
    print(f"Total documents: {len(full_with_folds):,}")
    print(f"Unique families/groups: {policy_df['__group_key'].nunique():,}")
    print(f"Number of folds: {args.n_folds}")

    for fold_id in range(args.n_folds):
        test_mask = full_with_folds["fold"] == fold_id
        test_out = full_with_folds.loc[test_mask, base_columns].reset_index(drop=True)
        train_out = full_with_folds.loc[~test_mask, base_columns].reset_index(drop=True)

        if train_out.empty or test_out.empty:
            raise RuntimeError(
                f"Fold {fold_id} has an empty train/test split. Reduce --n-folds or provide more data."
            )

        train_groups = set(policy_df.loc[~test_mask, "__group_key"].astype(str))
        test_groups = set(policy_df.loc[test_mask, "__group_key"].astype(str))
        overlap_groups = train_groups.intersection(test_groups)
        if overlap_groups:
            raise RuntimeError(
                f"Leakage detected in fold {fold_id}: group overlap between train and test."
            )

        train_path = kfold_dir / f"fold_{fold_id}_train.csv"
        test_path = kfold_dir / f"fold_{fold_id}_test.csv"
        train_out.to_csv(train_path, index=False)
        test_out.to_csv(test_path, index=False)

        print(
            f"Fold {fold_id}: train={len(train_out):,}, test={len(test_out):,}, "
            f"group-overlap={len(overlap_groups)}"
        )

    print(f"Assignments file: {assignment_path}")
    print(f"Per-fold directory: {kfold_dir}")


if __name__ == "__main__":
    main()
