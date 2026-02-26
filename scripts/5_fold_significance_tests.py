import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


METRIC_PREFIXES = ("hit@", "precision@", "ndcg@")


def safe_import_stats_tests() -> tuple[Optional[object], Optional[object]]:
    try:
        from scipy.stats import ttest_rel, wilcoxon  # type: ignore

        return ttest_rel, wilcoxon
    except Exception:
        return None, None


def parse_csv_list(raw: str) -> list[str]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    return sorted(set(values))


def infer_metric_columns(df: pd.DataFrame) -> list[str]:
    metric_cols = [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in METRIC_PREFIXES)
    ]
    if not metric_cols:
        raise ValueError(
            "No metric columns found. Expected columns starting with hit@, precision@, or ndcg@."
        )
    return sorted(metric_cols)


def validate_results_df(df: pd.DataFrame) -> None:
    required = {"fold", "method"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {missing}")


def build_method_pairs(
    methods: list[str],
    mode: str,
    method: Optional[str],
    baselines: list[str],
) -> list[tuple[str, str]]:
    method_set = set(methods)

    if mode == "method-vs-baselines":
        if not method:
            raise ValueError("`--method` is required for mode=method-vs-baselines")
        if method not in method_set:
            raise ValueError(f"Method `{method}` not found in results.")
        if not baselines:
            raise ValueError("`--baselines` is required for mode=method-vs-baselines")
        for baseline in baselines:
            if baseline not in method_set:
                raise ValueError(f"Baseline `{baseline}` not found in results.")
        return [(baseline, method) for baseline in baselines if baseline != method]

    if mode == "all-vs-baselines":
        if not baselines:
            raise ValueError("`--baselines` is required for mode=all-vs-baselines")
        for baseline in baselines:
            if baseline not in method_set:
                raise ValueError(f"Baseline `{baseline}` not found in results.")
        pairs: list[tuple[str, str]] = []
        for baseline in baselines:
            for candidate in methods:
                if candidate == baseline:
                    continue
                pairs.append((baseline, candidate))
        return pairs

    if mode == "pairwise":
        pairs: list[tuple[str, str]] = []
        for idx, left in enumerate(methods):
            for right in methods[idx + 1 :]:
                pairs.append((left, right))
        return pairs

    raise ValueError(f"Unknown mode: {mode}")


def compute_significance(
    results_df: pd.DataFrame,
    metric_cols: list[str],
    method_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    ttest_rel, wilcoxon = safe_import_stats_tests()
    rows: list[dict] = []

    for baseline_method, candidate_method in method_pairs:
        baseline_df = results_df[results_df["method"] == baseline_method]
        candidate_df = results_df[results_df["method"] == candidate_method]
        merged = baseline_df.merge(
            candidate_df,
            on="fold",
            suffixes=("_base", "_cand"),
            how="inner",
        )
        if merged.empty:
            continue

        for metric in metric_cols:
            base_values = merged[f"{metric}_base"].to_numpy(dtype=float)
            cand_values = merged[f"{metric}_cand"].to_numpy(dtype=float)

            p_t = np.nan
            p_w = np.nan
            if len(base_values) >= 2:
                if ttest_rel is not None:
                    try:
                        p_t = float(ttest_rel(cand_values, base_values).pvalue)
                    except Exception:
                        p_t = np.nan
                if wilcoxon is not None:
                    try:
                        p_w = float(wilcoxon(cand_values, base_values).pvalue)
                    except Exception:
                        p_w = np.nan

            rows.append(
                {
                    "baseline_method": baseline_method,
                    "candidate_method": candidate_method,
                    "metric": metric,
                    "n_folds": int(len(base_values)),
                    "baseline_mean": float(np.mean(base_values)),
                    "candidate_mean": float(np.mean(cand_values)),
                    "delta_candidate_minus_baseline": float(
                        np.mean(cand_values - base_values)
                    ),
                    "paired_ttest_pvalue": p_t,
                    "wilcoxon_pvalue": p_w,
                }
            )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired significance tests across folds using all_fold_results.csv "
            "and report p-values (paired t-test + Wilcoxon signed-rank)."
        )
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        required=False,
        help="Path to a single results CSV. If omitted, all results/*/summary_mean_metrics.csv will be used.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <results-csv-dir>/paired_significance_tests.csv",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["method-vs-baselines", "all-vs-baselines", "pairwise"],
        default="method-vs-baselines",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Target method for mode=method-vs-baselines (e.g., rag_summary_semantic).",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="",
        help="Comma-separated baseline methods.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help="Optional comma-separated metrics to test (defaults to all hit@/precision@/ndcg@ columns).",
    )
    return parser.parse_args()


def main() -> None:

    args = parse_args()

    # Auto-discover all summary_mean_metrics.csv if --results-csv is not provided
    if args.results_csv is None:
        import glob

        csv_paths = glob.glob("results/*/summary_mean_metrics.csv")
        if not csv_paths:
            raise FileNotFoundError(
                "No summary_mean_metrics.csv files found in results/*/"
            )
        dfs = []
        for path in csv_paths:
            df = pd.read_csv(path)
            df["experiment"] = Path(path).parent.name
            df = df.rename(columns={"window": "fold"})
            dfs.append(df)
        results_df = pd.concat(dfs, ignore_index=True)
        output_csv = Path("results/paired_significance_tests.csv")
    else:
        results_df = pd.read_csv(args.results_csv)
        if "window" in results_df.columns and "fold" not in results_df.columns:
            results_df = results_df.rename(columns={"window": "fold"})
        output_csv = args.output_csv
        if output_csv is None:
            output_csv = args.results_csv.parent / "paired_significance_tests.csv"

    # If 'method' column is missing, create it from 'backbone' and 'loss'
    if "method" not in results_df.columns:
        if "backbone" in results_df.columns and "loss" in results_df.columns:
            results_df["method"] = (
                results_df["backbone"].astype(str)
                + "|"
                + results_df["loss"].astype(str)
            )
        else:
            raise ValueError(
                "Cannot create 'method' column: missing 'backbone' or 'loss' columns."
            )

    validate_results_df(results_df)

    all_methods = sorted(results_df["method"].astype(str).unique().tolist())
    baselines = parse_csv_list(args.baselines)
    metric_cols = infer_metric_columns(results_df)
    requested_metrics = parse_csv_list(args.metrics)
    if requested_metrics:
        unknown = [m for m in requested_metrics if m not in metric_cols]
        if unknown:
            raise ValueError(f"Requested metrics not found in results CSV: {unknown}")
        metric_cols = requested_metrics

    method_pairs = build_method_pairs(
        methods=all_methods,
        mode=args.mode,
        method=args.method,
        baselines=baselines,
    )

    significance_df = compute_significance(
        results_df=results_df,
        metric_cols=metric_cols,
        method_pairs=method_pairs,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    significance_df.to_csv(output_csv, index=False)

    print(f"Methods found: {all_methods}")
    print(f"Compared pairs: {len(method_pairs)}")
    print(f"Metrics tested: {metric_cols}")
    print(f"Saved significance tests: {output_csv}")


if __name__ == "__main__":
    main()
