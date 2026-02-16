import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


try:
    from sentence_transformers import SentenceTransformer, util
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Missing dependency `sentence-transformers`. Install with `pip install sentence-transformers`."
    ) from exc

try:
    from rank_bm25 import BM25Okapi
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Missing dependency `rank-bm25`. Install with `pip install rank-bm25`."
    ) from exc


DEFAULT_POLICY_INPUT = Path("data/csv/group_kfold_assignments.csv")
DEFAULT_OUTPUT_DIR = Path("results/summary_only_baselines")
DEFAULT_K_VALUES = "1,5,10"
DEFAULT_ENCODER = "sentence-transformers/all-distilroberta-v1"


def safe_import_stats_tests() -> tuple[Optional[object], Optional[object]]:
    try:
        from scipy.stats import ttest_rel, wilcoxon  # type: ignore

        return ttest_rel, wilcoxon
    except Exception:
        return None, None


def parse_csv_ints(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    values = sorted(set(values))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_policy_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"fold", "Family Summary", "Document ID"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    out = df.copy()
    out["Family Summary"] = out["Family Summary"].apply(clean_text)
    out["Document ID"] = out["Document ID"].astype(str)
    out["fold"] = out["fold"].astype(int)
    out = out[(out["Family Summary"] != "") & (out["Document ID"] != "")]
    return out.reset_index(drop=True)


def aggregate_by_doc_max(
    scores: np.ndarray, doc_ids: list[str]
) -> tuple[np.ndarray, list[str]]:
    score_by_doc: dict[str, float] = {}
    for score, doc_id in zip(scores.tolist(), doc_ids):
        if doc_id not in score_by_doc:
            score_by_doc[doc_id] = float(score)
        else:
            score_by_doc[doc_id] = max(score_by_doc[doc_id], float(score))

    uniq_doc_ids = list(score_by_doc.keys())
    uniq_scores = np.asarray([score_by_doc[d] for d in uniq_doc_ids], dtype=np.float64)
    return uniq_scores, uniq_doc_ids


def rank_from_scores(scores: np.ndarray, doc_ids: list[str]) -> list[str]:
    agg_scores, agg_doc_ids = aggregate_by_doc_max(scores, doc_ids)
    if len(agg_doc_ids) == 0:
        return []
    order = np.argsort(-agg_scores)
    return [agg_doc_ids[i] for i in order]


class HumanSummaryBaseline:
    def __init__(self, summaries: list[str], doc_ids: list[str], model_name: str):
        self.model = SentenceTransformer(model_name)
        self.summaries = summaries
        self.doc_ids = doc_ids
        self.corpus_embeddings = self.model.encode(
            summaries,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

    def retrieve(self, query: str, k: int) -> list[str]:
        if not self.summaries:
            return []

        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        hits = util.semantic_search(
            query_embedding,
            self.corpus_embeddings,
            top_k=len(self.summaries),
        )

        scores = np.zeros(len(self.summaries), dtype=np.float64)
        for hit in hits[0]:
            scores[int(hit["corpus_id"])] = float(hit["score"])

        ranked = rank_from_scores(scores, self.doc_ids)
        return ranked[:k]


class ClimatePolicyRadarBaseline:
    def __init__(self, policy_documents: list[str], doc_ids: list[str]):
        self.docs = policy_documents
        self.doc_ids = doc_ids
        tokenized_corpus = [simple_tokenize(doc) for doc in policy_documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int) -> list[str]:
        if not self.docs:
            return []
        tokenized_query = simple_tokenize(query)
        scores = np.asarray(self.bm25.get_scores(tokenized_query), dtype=np.float64)
        ranked = rank_from_scores(scores, self.doc_ids)
        return ranked[:k]


def hit_at_k(ranked: list[str], target: str, k: int) -> float:
    return 1.0 if target in ranked[:k] else 0.0


def precision_at_k(ranked: list[str], target: str, k: int) -> float:
    return (1.0 / k) if target in ranked[:k] else 0.0


def ndcg_at_k(ranked: list[str], target: str, k: int) -> float:
    if target not in ranked[:k]:
        return 0.0
    rank = ranked[:k].index(target) + 1
    return 1.0 / np.log2(rank + 1)


def evaluate_fold(
    method_name: str,
    retriever: HumanSummaryBaseline | ClimatePolicyRadarBaseline,
    test_df: pd.DataFrame,
    k_values: list[int],
) -> dict[str, float]:
    records = {f"hit@{k}": [] for k in k_values}
    records.update({f"precision@{k}": [] for k in k_values})
    records.update({f"ndcg@{k}": [] for k in k_values})

    skipped = 0
    max_k = max(k_values)
    for _, row in test_df.iterrows():
        query = str(row["Family Summary"])
        target = str(row["Document ID"])
        if not query or not target:
            skipped += 1
            continue

        ranked = retriever.retrieve(query=query, k=max_k)
        if not ranked:
            skipped += 1
            continue

        for k in k_values:
            records[f"hit@{k}"].append(hit_at_k(ranked, target, k))
            records[f"precision@{k}"].append(precision_at_k(ranked, target, k))
            records[f"ndcg@{k}"].append(ndcg_at_k(ranked, target, k))

    metrics: dict[str, float] = {}
    for name, values in records.items():
        metrics[name] = float(np.mean(values)) if values else 0.0

    metrics["method"] = method_name
    metrics["queries_evaluated"] = float(max(len(test_df) - skipped, 0))
    metrics["queries_total"] = float(len(test_df))
    return metrics


def summarize_fold_metrics(results_df: pd.DataFrame, output_dir: Path) -> None:
    metric_cols = [
        col
        for col in results_df.columns
        if col.startswith("hit@")
        or col.startswith("precision@")
        or col.startswith("ndcg@")
    ]

    grouped = (
        results_df.groupby(["method"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values(["method"])
    )
    grouped.to_csv(output_dir / "summary_mean_metrics.csv", index=False)


def compute_significance(
    results_df: pd.DataFrame, baseline_method: str
) -> pd.DataFrame:
    ttest_rel, wilcoxon = safe_import_stats_tests()

    metric_cols = [
        col
        for col in results_df.columns
        if col.startswith("hit@")
        or col.startswith("precision@")
        or col.startswith("ndcg@")
    ]

    base = results_df[results_df["method"] == baseline_method].copy()
    if base.empty:
        raise ValueError(
            f"Baseline method `{baseline_method}` has no fold results for significance testing."
        )

    stat_rows: list[dict] = []
    candidates = sorted(results_df["method"].unique().tolist())

    for candidate_method in candidates:
        if candidate_method == baseline_method:
            continue

        candidate = results_df[results_df["method"] == candidate_method].copy()
        merged = base.merge(
            candidate, on="fold", suffixes=("_base", "_cand"), how="inner"
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

            stat_rows.append(
                {
                    "baseline_method": baseline_method,
                    "candidate_method": candidate_method,
                    "metric": metric,
                    "n_folds": len(base_values),
                    "baseline_mean": float(np.mean(base_values)),
                    "candidate_mean": float(np.mean(cand_values)),
                    "delta": float(np.mean(cand_values - base_values)),
                    "paired_ttest_pvalue": p_t,
                    "wilcoxon_pvalue": p_w,
                }
            )

    return pd.DataFrame(stat_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run two summary-only baselines over grouped folds: "
            "(1) semantic search with SentenceTransformer and "
            "(2) BM25 keyword retrieval."
        )
    )
    parser.add_argument("--policy-input", type=Path, default=DEFAULT_POLICY_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-values", type=str, default=DEFAULT_K_VALUES)
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument(
        "--significance-baseline", type=str, default="human_summary_semantic"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    k_values = parse_csv_ints(args.k_values)
    if any(k <= 0 for k in k_values):
        raise ValueError("All k-values must be positive")

    policy_df = load_policy_df(args.policy_input)
    fold_ids = sorted(policy_df["fold"].unique().tolist())
    if args.fold is not None:
        if args.fold not in fold_ids:
            raise ValueError(f"Fold {args.fold} does not exist in {args.policy_input}")
        fold_ids = [args.fold]

    all_results: list[dict] = []
    for fold_id in fold_ids:
        train_df = policy_df[policy_df["fold"] != fold_id][
            ["Family Summary", "Document ID"]
        ].copy()
        test_df = policy_df[policy_df["fold"] == fold_id][
            ["Family Summary", "Document ID"]
        ].copy()

        train_df = train_df.drop_duplicates().reset_index(drop=True)
        test_df = test_df.drop_duplicates().reset_index(drop=True)

        summaries = train_df["Family Summary"].tolist()
        doc_ids = train_df["Document ID"].astype(str).tolist()

        semantic = HumanSummaryBaseline(
            summaries=summaries, doc_ids=doc_ids, model_name=args.encoder
        )
        bm25 = ClimatePolicyRadarBaseline(policy_documents=summaries, doc_ids=doc_ids)

        for method_name, retriever in [
            ("human_summary_semantic", semantic),
            ("climate_policy_radar_bm25", bm25),
        ]:
            result = evaluate_fold(
                method_name=method_name,
                retriever=retriever,
                test_df=test_df,
                k_values=k_values,
            )
            result["fold"] = int(fold_id)
            all_results.append(result)

            print(
                f"Finished fold={fold_id}, method={method_name}, "
                f"hit@5={result.get('hit@5', 0.0):.4f}, "
                f"ndcg@5={result.get('ndcg@5', 0.0):.4f}, "
                f"precision@5={result.get('precision@5', 0.0):.4f}"
            )

    results_df = pd.DataFrame(all_results).sort_values(["fold", "method"])
    results_path = args.output_dir / "all_fold_results.csv"
    results_df.to_csv(results_path, index=False)

    summarize_fold_metrics(results_df, args.output_dir)

    significance_df = compute_significance(
        results_df=results_df,
        baseline_method=args.significance_baseline,
    )
    significance_path = args.output_dir / "paired_significance_tests.csv"
    significance_df.to_csv(significance_path, index=False)

    metadata = {
        "task": "summary_only_retrieval_baselines",
        "folds": fold_ids,
        "k_values": k_values,
        "encoder": args.encoder,
        "policy_input": str(args.policy_input),
        "methods": ["human_summary_semantic", "climate_policy_radar_bm25"],
        "metrics": ["Hit@k", "Precision@k", "NDCG@k"],
        "results_csv": str(results_path),
        "significance_baseline": args.significance_baseline,
        "significance_csv": str(significance_path),
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved results: {results_path}")
    print(f"Saved significance tests: {significance_path}")


if __name__ == "__main__":
    main()
