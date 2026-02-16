import argparse
import json
import re
from html import unescape
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from utils.data import policy_sectors
except ModuleNotFoundError:
    from scripts.utils.data import policy_sectors


DEFAULT_POLICY_INPUT = Path("data/csv/group_kfold_assignments.csv")
DEFAULT_TIME_SERIES_DIR = Path("data/time_series/kfold")
DEFAULT_OUTPUT_DIR = Path("data/model_input/kfold")
DEFAULT_WINDOWS = "1,2,5,10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare policy-text + time-series training inputs from grouped k-fold data "
            "and generate ablation sets for multiple history windows."
        )
    )
    parser.add_argument("--policy-input", type=Path, default=DEFAULT_POLICY_INPUT)
    parser.add_argument("--time-series-dir", type=Path, default=DEFAULT_TIME_SERIES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--windows", type=str, default=DEFAULT_WINDOWS)
    parser.add_argument("--negative-samples", type=int, default=1)
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Process only one fold. By default, all folds in policy-input are processed.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_windows(raw: str) -> list[int]:
    windows = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("All history window values must be positive integers")
        windows.append(value)
    unique_sorted = sorted(set(windows))
    if not unique_sorted:
        raise ValueError("At least one history window must be provided")
    return unique_sorted


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned.strip("_") or "UNKNOWN_COUNTRY"


def split_sector_values(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    sectors = [part.strip() for part in value.split(";") if part.strip()]
    return [sector for sector in sectors if sector != "Other"]


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", value)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_policy_df(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "Family Summary",
        "Sector",
        "Geography",
        "Last event in timeline",
    ]
    for column in required_cols:
        if column not in df.columns:
            raise ValueError(f"Missing required column in policy input: {column}")

    out = df.copy()
    out["Family Summary"] = out["Family Summary"].apply(clean_text)
    out = out[out["Family Summary"].str.strip() != ""]

    out["Sector"] = out["Sector"].apply(split_sector_values)
    out = out[out["Sector"].map(len) > 0]

    out["Geography"] = out["Geography"].astype(str).str.strip()
    out = out[out["Geography"] != ""]

    out["Last event in timeline"] = pd.to_datetime(
        out["Last event in timeline"], errors="coerce"
    )
    out = out.dropna(subset=["Last event in timeline"])

    if "Document ID" not in out.columns:
        out["Document ID"] = ""

    out = out.reset_index(drop=True)
    return out


def read_time_series_fold(fold_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dir = fold_dir / "train"
    test_dir = fold_dir / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Fold directory must contain train/ and test/: {fold_dir}"
        )

    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    for csv_path in sorted(train_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if "country" not in df.columns or "year" not in df.columns:
            continue
        train_frames.append(df)

    for csv_path in sorted(test_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if "country" not in df.columns or "year" not in df.columns:
            continue
        test_frames.append(df)

    if not train_frames:
        raise RuntimeError(f"No train country CSVs found for fold at {train_dir}")
    if not test_frames:
        raise RuntimeError(f"No test country CSVs found for fold at {test_dir}")

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)
    return train_df, test_df


def align_and_scale_time_series(
    train_ts: pd.DataFrame,
    test_ts: pd.DataFrame,
    fold_output_dir: Path,
) -> pd.DataFrame:
    all_columns = sorted(set(train_ts.columns).union(test_ts.columns))

    def align_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for column in all_columns:
            if column not in out.columns:
                out[column] = 0
        out = out[all_columns].fillna(0)
        return out

    train_aligned = align_columns(train_ts)
    test_aligned = align_columns(test_ts)

    numeric_columns = train_aligned.select_dtypes(
        include=["number"]
    ).columns.difference(["year"])

    scaler = StandardScaler()
    if len(numeric_columns) > 0:
        dtype_map = {column: "float64" for column in numeric_columns}
        train_aligned = train_aligned.astype(dtype_map)
        test_aligned = test_aligned.astype(dtype_map)
        scaler.fit(train_aligned[numeric_columns])
        train_aligned.loc[:, numeric_columns] = scaler.transform(
            train_aligned[numeric_columns]
        )
        test_aligned.loc[:, numeric_columns] = scaler.transform(
            test_aligned[numeric_columns]
        )

    fold_output_dir.mkdir(parents=True, exist_ok=True)
    train_aligned.to_csv(fold_output_dir / "scaled_train_time_series.csv", index=False)
    test_aligned.to_csv(fold_output_dir / "scaled_test_time_series.csv", index=False)
    joblib.dump(scaler, fold_output_dir / "scaler.pkl")

    combined = pd.concat([train_aligned, test_aligned], ignore_index=True)
    combined = combined.sort_values(["country", "year"]).reset_index(drop=True)
    return combined


def sector_one_hot(sectors: list[str]) -> list[int]:
    encoding = np.zeros(len(policy_sectors), dtype=int)
    for sector in sectors:
        if sector in policy_sectors:
            encoding[policy_sectors.index(sector)] = 1
    return encoding.tolist()


def build_window_tensor(
    scaled_dataframe: pd.DataFrame,
    country: str,
    event_year: int,
    window: int,
    feature_columns: list[str],
) -> np.ndarray:
    start_year = event_year - window + 1
    expected_years = pd.DataFrame({"year": list(range(start_year, event_year + 1))})

    country_rows = scaled_dataframe.loc[
        scaled_dataframe["country"] == country, ["year", *feature_columns]
    ].copy()
    if country_rows.empty:
        return np.zeros((window, len(feature_columns)), dtype=float)

    country_rows["year"] = pd.to_numeric(country_rows["year"], errors="coerce")
    country_rows = country_rows.dropna(subset=["year"])
    country_rows["year"] = country_rows["year"].astype(int)

    merged = expected_years.merge(country_rows, on="year", how="left")
    merged = merged.drop(columns=["year"]).fillna(0)
    return merged.to_numpy(dtype=float)


def process_split_for_window(
    df: pd.DataFrame,
    scaled_dataframe: pd.DataFrame,
    window: int,
    mode: str,
    feature_columns: list[str],
) -> list[dict]:
    records: list[dict] = []

    for _, row in df.iterrows():
        history = build_window_tensor(
            scaled_dataframe=scaled_dataframe,
            country=row["Geography"],
            event_year=row["Last event in timeline"].year,
            window=window,
            feature_columns=feature_columns,
        )
        encoded_sector = sector_one_hot(row["Sector"])

        item = {
            "anchor_summary": row["Family Summary"],
            "positive_time_series": history.tolist(),
            "positive_sector": encoded_sector,
        }

        if mode == "test":
            item.update(
                {
                    "doc_id": str(row.get("Document ID", "")),
                    "sector": row["Sector"],
                    "country": row["Geography"],
                }
            )

        records.append(item)

    return records


def generate_negative_pairs(
    train_df: pd.DataFrame,
    scaled_dataframe: pd.DataFrame,
    positives: list[dict],
    window: int,
    negative_samples: int,
    rng: np.random.Generator,
    feature_columns: list[str],
) -> list[dict]:
    pairs: list[dict] = []

    for pos in positives:
        anchor_summary = pos["anchor_summary"]
        candidate_df = train_df[train_df["Family Summary"] != anchor_summary]
        if candidate_df.empty:
            continue

        sample_count = min(negative_samples, len(candidate_df))
        sampled_idx = rng.choice(
            candidate_df.index.to_numpy(), size=sample_count, replace=False
        )
        sampled_rows = candidate_df.loc[sampled_idx]

        for _, sampled in sampled_rows.iterrows():
            negative_history = build_window_tensor(
                scaled_dataframe=scaled_dataframe,
                country=sampled["Geography"],
                event_year=sampled["Last event in timeline"].year,
                window=window,
                feature_columns=feature_columns,
            )

            pairs.append(
                {
                    "anchor_summary": anchor_summary,
                    "positive_time_series": pos["positive_time_series"],
                    "positive_sector": pos["positive_sector"],
                    "negative_time_series": negative_history.tolist(),
                    "negative_sector": sector_one_hot(sampled["Sector"]),
                }
            )

    return pairs


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_fold_ids(
    policy_df: pd.DataFrame, requested_fold: Optional[int]
) -> list[int]:
    if "fold" not in policy_df.columns:
        raise ValueError(
            "Expected grouped K-fold assignments with a 'fold' column. "
            "Run scripts/0_split_test_set.py first."
        )

    fold_ids = sorted(policy_df["fold"].dropna().astype(int).unique().tolist())
    if not fold_ids:
        raise ValueError("No fold values found in policy input")

    if requested_fold is None:
        return fold_ids

    if requested_fold not in fold_ids:
        raise ValueError(
            f"Requested fold {requested_fold} not found in available folds: {fold_ids}"
        )
    return [requested_fold]


def process_fold(
    policy_df: pd.DataFrame,
    fold_id: int,
    args: argparse.Namespace,
    windows: list[int],
) -> dict[str, int]:
    fold_ts_dir = args.time_series_dir / f"fold_{fold_id}"
    fold_output_dir = args.output_dir / f"fold_{fold_id}"

    train_policy = clean_policy_df(policy_df[policy_df["fold"] != fold_id])
    test_policy = clean_policy_df(policy_df[policy_df["fold"] == fold_id])

    train_ts, test_ts = read_time_series_fold(fold_ts_dir)
    scaled_dataframe = align_and_scale_time_series(train_ts, test_ts, fold_output_dir)

    available_countries = set(scaled_dataframe["country"].dropna().astype(str).unique())
    train_policy = train_policy[train_policy["Geography"].isin(available_countries)]
    test_policy = test_policy[test_policy["Geography"].isin(available_countries)]

    if train_policy.empty or test_policy.empty:
        raise RuntimeError(
            f"Fold {fold_id}: no policy rows after matching available time-series countries"
        )

    feature_columns = [
        col for col in scaled_dataframe.columns if col not in {"country", "year"}
    ]
    rng = np.random.default_rng(args.seed + fold_id)

    summary = {
        "fold": int(fold_id),
        "train_policy_rows": int(len(train_policy)),
        "test_policy_rows": int(len(test_policy)),
        "feature_count": int(len(feature_columns)),
        "windows": {},
    }

    for window in windows:
        window_dir = fold_output_dir / f"window_{window}"
        window_dir.mkdir(parents=True, exist_ok=True)

        positive_train = process_split_for_window(
            train_policy,
            scaled_dataframe,
            window=window,
            mode="train",
            feature_columns=feature_columns,
        )
        test_records = process_split_for_window(
            test_policy,
            scaled_dataframe,
            window=window,
            mode="test",
            feature_columns=feature_columns,
        )

        train_pairs = generate_negative_pairs(
            train_df=train_policy,
            scaled_dataframe=scaled_dataframe,
            positives=positive_train,
            window=window,
            negative_samples=args.negative_samples,
            rng=rng,
            feature_columns=feature_columns,
        )

        write_jsonl(window_dir / "train.jsonl", train_pairs)
        with (window_dir / "test.json").open("w", encoding="utf-8") as handle:
            json.dump(test_records, handle, ensure_ascii=False, indent=2)

        summary["windows"][str(window)] = {
            "train_pairs": int(len(train_pairs)),
            "test_rows": int(len(test_records)),
        }

        print(
            f"Fold {fold_id}, window {window}: "
            f"train_pairs={len(train_pairs)}, test_rows={len(test_records)}"
        )

    with (fold_output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    windows = parse_windows(args.windows)

    policy_df = pd.read_csv(args.policy_input)
    fold_ids = resolve_fold_ids(policy_df, args.fold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_manifests = [
        process_fold(policy_df, fold_id, args, windows) for fold_id in fold_ids
    ]

    summary_path = args.output_dir / "preparation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"folds": all_manifests}, handle, indent=2)

    print(f"Saved preparation summary: {summary_path}")


if __name__ == "__main__":
    main()
