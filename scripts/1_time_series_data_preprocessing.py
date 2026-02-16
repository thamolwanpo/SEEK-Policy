import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_OWID_DIR = Path("data/owid")
DEFAULT_POLICY_INPUT = Path("data/csv/group_kfold_assignments.csv")
DEFAULT_OUTPUT_DIR = Path("data/time_series")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess OWID time-series data with leakage-safe train-only feature fitting "
            "for grouped K-fold evaluation."
        )
    )
    parser.add_argument("--owid-dir", type=Path, default=DEFAULT_OWID_DIR)
    parser.add_argument("--policy-input", type=Path, default=DEFAULT_POLICY_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-year", type=int, default=1970)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--missing-threshold", type=float, default=0.2)
    parser.add_argument("--correlation-threshold", type=float, default=0.7)
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Process only one fold. By default, all folds in policy-input are processed.",
    )
    return parser.parse_args()


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "UNKNOWN_COUNTRY"


def discover_owid_tables(owid_dir: Path) -> list[tuple[str, Path]]:
    if not owid_dir.exists():
        raise FileNotFoundError(f"OWID directory not found: {owid_dir}")

    tables: list[tuple[str, Path]] = []
    for meta_path in sorted(owid_dir.rglob("*.meta.json")):
        csv_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".csv"))
        if not csv_path.exists():
            continue
        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        title = metadata.get("title")
        if not title:
            dataset_obj = metadata.get("dataset", {})
            title = dataset_obj.get("title")
        if not title:
            title = csv_path.stem
        tables.append((str(title), csv_path))

    if not tables:
        raise RuntimeError(f"No OWID table metadata found under {owid_dir}")
    return tables


def standardize_table(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    if "country" not in df.columns and "location" in df.columns:
        df = df.rename(columns={"location": "country"})
    elif "country" in df.columns and "location" in df.columns:
        df = df.drop(columns=["location"])

    if "country" not in df.columns or "year" not in df.columns:
        return pd.DataFrame(columns=["country", "year"])

    df = df.copy()
    df["country"] = df["country"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df[(df["year"] >= start_year) & (df["year"] <= end_year)]


def select_numeric_features_train_only(
    train_df: pd.DataFrame,
    missing_threshold: float,
    correlation_threshold: float,
) -> list[str]:
    numeric_df = train_df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.drop(columns=["year"], errors="ignore")
    if numeric_df.empty:
        return []

    non_constant = numeric_df.nunique(dropna=True)
    numeric_df = numeric_df.loc[:, non_constant > 1]
    if numeric_df.empty:
        return []

    min_non_missing = int(np.ceil(len(numeric_df) * (1 - missing_threshold)))
    numeric_df = numeric_df.dropna(axis=1, thresh=min_non_missing)
    if numeric_df.empty:
        return []

    corr = numeric_df.corr().abs()
    if corr.empty:
        return numeric_df.columns.tolist()

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [
        col for col in upper.columns if (upper[col] > correlation_threshold).any()
    ]
    selected = [col for col in numeric_df.columns if col not in to_drop]
    return selected


def deduplicate_country_year(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["__nn"] = df.notna().sum(axis=1)
    df = df.sort_values("__nn", ascending=False).drop_duplicates(
        ["country", "year"], keep="first"
    )
    return df.drop(columns=["__nn"])


def impute_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns.difference(["year"])
    if len(numeric_cols) == 0:
        return out
    out[numeric_cols] = out[numeric_cols].interpolate(method="linear", axis=0)
    out[numeric_cols] = out[numeric_cols].bfill().ffill()
    return out


def build_country_frame(
    country: str,
    year_range: pd.DataFrame,
    selected_tables: Iterable[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    merged = year_range.copy()
    merged["country"] = country

    for table_key, table_df in selected_tables:
        country_slice = table_df.loc[table_df["country"] == country].copy()
        if country_slice.empty:
            continue
        rename_map = {
            col: f"{table_key}__{col}"
            for col in country_slice.columns
            if col not in {"country", "year"}
        }
        country_slice = country_slice.rename(columns=rename_map)
        country_slice = country_slice.drop(columns=["country"])
        merged = merged.merge(country_slice, on="year", how="left")

    merged = merged.sort_values("year").reset_index(drop=True)
    return impute_numeric(merged)


def process_fold(
    policy_df: pd.DataFrame,
    fold_id: int,
    tables: list[tuple[str, Path]],
    args: argparse.Namespace,
) -> dict[str, int]:
    if "fold" not in policy_df.columns:
        raise ValueError("policy-input must include a 'fold' column for k-fold mode")
    if "Geography" not in policy_df.columns:
        raise ValueError("policy-input must include a 'Geography' column")

    train_df = policy_df[policy_df["fold"] != fold_id]
    test_df = policy_df[policy_df["fold"] == fold_id]
    train_countries = set(train_df["Geography"].dropna().astype(str).str.strip())
    test_countries = set(test_df["Geography"].dropna().astype(str).str.strip())
    all_countries = sorted(train_countries.union(test_countries))

    if not train_countries or not test_countries:
        raise ValueError(
            f"Fold {fold_id} has empty train or test policy countries. "
            "Check fold assignments before preprocessing."
        )

    selected_tables: list[tuple[str, pd.DataFrame]] = []
    total_feature_count = 0

    for table_idx, (title, csv_path) in enumerate(tables):
        raw = pd.read_csv(csv_path)
        standardized = standardize_table(raw, args.start_year, args.end_year)
        if standardized.empty:
            continue

        standardized = standardized[standardized["country"].isin(all_countries)]
        if standardized.empty:
            continue

        standardized = deduplicate_country_year(standardized)
        train_slice = standardized[standardized["country"].isin(train_countries)]
        if train_slice.empty:
            continue

        selected_cols = select_numeric_features_train_only(
            train_slice,
            missing_threshold=args.missing_threshold,
            correlation_threshold=args.correlation_threshold,
        )
        if not selected_cols:
            continue

        keep_cols = ["country", "year", *selected_cols]
        filtered = standardized[keep_cols].copy()
        table_key = f"t{table_idx:03d}"
        total_feature_count += len(selected_cols)
        selected_tables.append((table_key, filtered))

    if not selected_tables:
        raise RuntimeError(
            f"No usable time-series features were selected for fold {fold_id}."
        )

    fold_dir = args.output_dir / "kfold" / f"fold_{fold_id}"
    train_out_dir = fold_dir / "train"
    test_out_dir = fold_dir / "test"
    train_out_dir.mkdir(parents=True, exist_ok=True)
    test_out_dir.mkdir(parents=True, exist_ok=True)

    year_range = pd.DataFrame({"year": list(range(args.start_year, args.end_year + 1))})
    rows_written = 0
    for country in all_countries:
        country_frame = build_country_frame(country, year_range, selected_tables)
        target_dir = train_out_dir if country in train_countries else test_out_dir
        out_name = f"{sanitize_filename(country)}.csv"
        country_frame.to_csv(target_dir / out_name, index=False)
        rows_written += len(country_frame)

    manifest = {
        "fold": int(fold_id),
        "train_country_count": int(len(train_countries)),
        "test_country_count": int(len(test_countries)),
        "selected_table_count": int(len(selected_tables)),
        "selected_feature_count": int(total_feature_count),
        "start_year": int(args.start_year),
        "end_year": int(args.end_year),
    }
    with (fold_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(
        f"Fold {fold_id}: train_countries={len(train_countries)}, "
        f"test_countries={len(test_countries)}, tables={len(selected_tables)}, "
        f"features={total_feature_count}, rows={rows_written}"
    )
    return manifest


def main() -> None:
    args = parse_args()
    policy_df = pd.read_csv(args.policy_input)
    tables = discover_owid_tables(args.owid_dir)

    if "fold" not in policy_df.columns:
        raise ValueError(
            "Expected grouped K-fold assignments in policy-input with a 'fold' column. "
            "Run scripts/0_split_test_set.py first."
        )

    available_folds = [
        int(value) for value in sorted(policy_df["fold"].dropna().astype(int).unique())
    ]
    if not available_folds:
        raise ValueError("No fold values found in policy-input")

    if args.fold is not None:
        if args.fold not in available_folds:
            raise ValueError(
                f"Requested fold {args.fold} not in available folds: {available_folds}"
            )
        fold_ids = [args.fold]
    else:
        fold_ids = available_folds

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = [process_fold(policy_df, fold_id, tables, args) for fold_id in fold_ids]

    summary_path = args.output_dir / "kfold" / "preprocessing_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"folds": manifests}, handle, indent=2)

    print(f"Saved preprocessing summary: {summary_path}")


if __name__ == "__main__":
    main()
