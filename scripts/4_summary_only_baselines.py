import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Optional, TypeVar

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


try:
    from langchain_openai import OpenAIEmbeddings
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Missing dependency `langchain-openai`. Install with `pip install langchain-openai`."
    ) from exc

try:
    from rank_bm25 import BM25Okapi
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Missing dependency `rank-bm25`. Install with `pip install rank-bm25`."
    ) from exc

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

try:
    from dotenv import find_dotenv, load_dotenv
except Exception:  # pragma: no cover - optional dependency
    find_dotenv = None
    load_dotenv = None


DEFAULT_POLICY_INPUT = Path("data/csv/group_kfold_assignments.csv")
DEFAULT_OUTPUT_DIR = Path("results/summary_only_baselines")
DEFAULT_METRIC_TOP_K_VALUES = "1,5,10"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_EMBEDDING_BATCH_SIZE = 128
DEFAULT_CHUNK_VECTORDB_DIR = Path("data/vectorstore/policy_chunks_chroma")
DEFAULT_CHUNK_VECTORDB_COLLECTION = "policy_chunks_openai"

T = TypeVar("T")


def with_progress(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    unit: str = "it",
) -> Iterable[T]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit)


def ensure_openai_api_key_loaded() -> None:
    if os.getenv("OPENAI_API_KEY", "").strip():
        return

    if load_dotenv is not None:
        dotenv_path = ""
        if find_dotenv is not None:
            dotenv_path = find_dotenv(filename=".env", usecwd=True)

        if dotenv_path:
            load_dotenv(dotenv_path=dotenv_path, override=False)
        else:
            load_dotenv(override=False)

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError(
            "OPENAI_API_KEY is not set. Add OPENAI_API_KEY=... to your shell env or .env in the project root."
        )


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


def load_vectordb_manifest(vectordb_dir: Path) -> dict:
    manifest_path = vectordb_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def metadata_get_str(metadata: dict, keys: list[str]) -> str:
    for key in keys:
        if key not in metadata:
            continue
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def metadata_get_int(metadata: dict, keys: list[str]) -> Optional[int]:
    for key in keys:
        if key not in metadata:
            continue
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def resolve_chunk_collection_name(
    vectordb_dir: Path,
    requested_collection_name: str,
) -> str:
    requested = str(requested_collection_name).strip()
    manifest = load_vectordb_manifest(vectordb_dir)
    manifest_collection = str(manifest.get("collection", "")).strip()

    if requested and requested.lower() not in {"auto", "manifest"}:
        return requested
    if manifest_collection:
        return manifest_collection
    if requested:
        return requested
    return DEFAULT_CHUNK_VECTORDB_COLLECTION


def load_chunk_corpus_from_vectordb(
    vectordb_dir: Path,
    collection_name: str,
    chunk_fold_ids: list[int],
) -> tuple[pd.DataFrame, Optional[np.ndarray]]:
    try:
        from langchain_chroma import Chroma
    except Exception:
        from langchain_community.vectorstores import Chroma

    if not vectordb_dir.exists():
        raise FileNotFoundError(
            f"Chunk vector DB directory not found: {vectordb_dir}. "
            "Run scripts/3_build_chunk_vectordb.py first."
        )

    resolved_collection = resolve_chunk_collection_name(
        vectordb_dir=vectordb_dir,
        requested_collection_name=collection_name,
    )

    store = Chroma(
        collection_name=resolved_collection,
        embedding_function=None,
        persist_directory=str(vectordb_dir),
    )
    page_size = 1000
    offset = 0
    documents: list[str] = []
    metadatas: list[dict] = []
    embeddings: list[object] = []

    while True:
        payload = store.get(
            include=["documents", "metadatas", "embeddings"],
            limit=page_size,
            offset=offset,
        )
        page_documents = payload.get("documents", []) or []
        page_metadatas = payload.get("metadatas", []) or []
        page_embeddings = payload.get("embeddings", None)
        if page_embeddings is None:
            page_embeddings = []
        elif isinstance(page_embeddings, np.ndarray):
            page_embeddings = page_embeddings.tolist()

        if not page_documents:
            break

        if len(page_embeddings) != len(page_documents):
            page_embeddings = [None] * len(page_documents)

        documents.extend(page_documents)
        metadatas.extend(page_metadatas)
        embeddings.extend(page_embeddings)

        if len(page_documents) < page_size:
            break
        offset += page_size

    if not documents or not metadatas:
        raise RuntimeError(
            f"No chunk entries found in vector DB collection `{resolved_collection}` at {vectordb_dir}."
        )

    fold_set = set(int(fold_id) for fold_id in chunk_fold_ids)
    rows: list[dict[str, str | int]] = []
    row_embeddings: list[object] = []
    for chunk_text, metadata, embedding in zip(documents, metadatas, embeddings):
        metadata = metadata or {}
        fold_value = metadata_get_int(metadata, ["fold", "Fold"])
        doc_id = metadata_get_str(
            metadata,
            ["document_id", "Document ID", "doc_id", "documentId"],
        )
        chunk_id = metadata_get_str(metadata, ["chunk_id", "chunkId"])
        source = metadata_get_str(metadata, ["source", "file_path", "path"])
        cleaned_chunk = clean_text(chunk_text)

        if fold_value is None or not doc_id or not cleaned_chunk:
            continue

        if fold_value not in fold_set:
            continue

        rows.append(
            {
                "fold": fold_value,
                "Document ID": doc_id,
                "chunk_text": cleaned_chunk,
                "chunk_id": chunk_id,
                "source": source,
            }
        )
        row_embeddings.append(embedding)

    dedup_rows: list[dict[str, str | int]] = []
    dedup_embeddings: list[object] = []
    seen_keys: set[tuple[object, ...]] = set()
    for row, embedding in zip(rows, row_embeddings):
        key = (
            row["fold"],
            row["Document ID"],
            row["chunk_id"],
            row["chunk_text"],
            row["source"],
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        dedup_rows.append(row)
        dedup_embeddings.append(embedding)

    chunk_df = pd.DataFrame(dedup_rows)
    if chunk_df.empty:
        raise RuntimeError(
            "Chunk corpus from vector DB is empty for selected training folds. "
            "Check stored fold metadata and train/test split settings."
        )

    precomputed_embeddings: Optional[np.ndarray] = None
    if dedup_embeddings and all(
        isinstance(embedding, (list, tuple)) and len(embedding) > 0
        for embedding in dedup_embeddings
    ):
        try:
            precomputed_embeddings = np.asarray(dedup_embeddings, dtype=np.float64)
        except Exception:
            precomputed_embeddings = None

    return chunk_df.reset_index(drop=True), precomputed_embeddings


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


class HumanSummaryChunkSemanticRetriever:
    def __init__(
        self,
        chunk_texts: list[str],
        model_name: str,
        embedding_batch_size: int,
        precomputed_chunk_embeddings: Optional[np.ndarray] = None,
    ):
        ensure_openai_api_key_loaded()
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be > 0")

        self.embeddings = OpenAIEmbeddings(model=model_name)
        self.chunk_texts = chunk_texts
        if precomputed_chunk_embeddings is not None:
            print(
                "Using precomputed chunk embeddings with shape:",
                precomputed_chunk_embeddings.shape,
            )
            if precomputed_chunk_embeddings.shape[0] != len(chunk_texts):
                raise ValueError(
                    "precomputed_chunk_embeddings row count must match chunk_texts length"
                )
            self.chunk_embeddings = np.asarray(
                precomputed_chunk_embeddings, dtype=np.float64
            )
        else:
            print("Computing chunk embeddings for", len(chunk_texts), "chunks...")
            vectors: list[list[float]] = []
            if chunk_texts:
                batch_starts = range(0, len(chunk_texts), embedding_batch_size)
                for start in batch_starts:
                    batch = chunk_texts[start : start + embedding_batch_size]
                    vectors.extend(self.embeddings.embed_documents(batch))

            self.chunk_embeddings = np.asarray(vectors, dtype=np.float64)

    def score_chunks(self, query: str) -> np.ndarray:
        if not self.chunk_texts:
            return np.zeros(0, dtype=np.float64)
        query_embedding = np.asarray(
            [self.embeddings.embed_query(query)], dtype=np.float64
        )
        return cosine_similarity(query_embedding, self.chunk_embeddings)[0].astype(
            np.float64
        )


class ChunkBM25Retriever:
    def __init__(self, chunk_texts: list[str]):
        self.chunk_texts = chunk_texts
        tokenized_corpus = [simple_tokenize(doc) for doc in chunk_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def score_chunks(self, query: str) -> np.ndarray:
        if not self.chunk_texts:
            return np.zeros(0, dtype=np.float64)
        tokenized_query = simple_tokenize(query)
        return np.asarray(self.bm25.get_scores(tokenized_query), dtype=np.float64)


def hit_at_k(ranked: list[str], target: str, k: int) -> float:
    return 1.0 if target in ranked[:k] else 0.0


def precision_at_k(ranked: list[str], target: str, k: int) -> float:
    return (1.0 / k) if target in ranked[:k] else 0.0


def ndcg_at_k(ranked: list[str], target: str, k: int) -> float:
    if target not in ranked[:k]:
        return 0.0
    rank = ranked[:k].index(target) + 1
    return 1.0 / np.log2(rank + 1)


def mrr_at_k(ranked: list[str], target: str, k: int) -> float:
    if target not in ranked[:k]:
        return 0.0
    rank = ranked[:k].index(target) + 1
    return 1.0 / rank


def ranked_doc_ids_from_scores(
    scores: np.ndarray, corpus_doc_ids: list[str]
) -> list[str]:
    if len(corpus_doc_ids) == 0:
        return []
    agg_scores, agg_doc_ids = aggregate_by_doc_max(scores, corpus_doc_ids)
    order = np.argsort(-agg_scores)
    return [agg_doc_ids[i] for i in order]


def evaluate_fold(
    fold_id: int,
    method_name: str,
    retriever: object,
    test_df: pd.DataFrame,
    chunk_corpus_df: pd.DataFrame,
    k_values: list[int],
    save_retrieval_traces: bool,
    retrieval_trace_top_k: int,
    retrieval_trace_output_path: Optional[Path],
    show_progress: bool,
) -> dict[str, float]:
    records = {f"hit@{k}": [] for k in k_values}
    records.update({f"precision@{k}": [] for k in k_values})
    records.update({f"ndcg@{k}": [] for k in k_values})
    records.update({f"mrr@{k}": [] for k in k_values})

    corpus_doc_ids = chunk_corpus_df["Document ID"].astype(str).tolist()
    corpus_chunk_texts = chunk_corpus_df["chunk_text"].astype(str).tolist()
    corpus_chunk_ids = chunk_corpus_df["chunk_id"].astype(str).tolist()
    corpus_chunk_sources = chunk_corpus_df["source"].astype(str).tolist()
    trace_top_k = retrieval_trace_top_k if retrieval_trace_top_k > 0 else max(k_values)
    retrieval_trace_rows: list[dict] = []

    skipped = 0
    query_iter: Iterable[tuple[int, pd.Series]] = test_df.iterrows()
    if show_progress:
        query_iter = with_progress(
            query_iter,
            total=len(test_df),
            desc=f"fold {fold_id} | {method_name} queries",
            unit="query",
        )

    for query_index, row in query_iter:
        query = str(row["Family Summary"])
        target = str(row["Document ID"])
        if not query or not target:
            skipped += 1
            continue

        scores = retriever.score_chunks(query)
        if len(scores) != len(corpus_doc_ids):
            raise RuntimeError(
                f"Retriever `{method_name}` returned {len(scores)} chunk scores, "
                f"expected {len(corpus_doc_ids)} to match chunk corpus size."
            )

        ranked = ranked_doc_ids_from_scores(scores, corpus_doc_ids)
        if not ranked:
            skipped += 1
            continue

        if save_retrieval_traces and retrieval_trace_output_path is not None:
            chunk_rank_order = np.argsort(-scores)
            max_rows = min(trace_top_k, len(chunk_rank_order))
            for rank in range(max_rows):
                chunk_idx = int(chunk_rank_order[rank])
                retrieved_doc_id = str(corpus_doc_ids[chunk_idx])
                retrieval_trace_rows.append(
                    {
                        "method": method_name,
                        "query_index": int(query_index),
                        "query_summary": query,
                        "target_doc_id": target,
                        "retrieved_rank": rank + 1,
                        "retrieved_doc_id": retrieved_doc_id,
                        "retrieved_score": float(scores[chunk_idx]),
                        "is_relevant_doc": float(retrieved_doc_id == target),
                        "retrieved_chunk_id": corpus_chunk_ids[chunk_idx],
                        "retrieved_chunk_source": corpus_chunk_sources[chunk_idx],
                        "retrieved_chunk_text": corpus_chunk_texts[chunk_idx],
                    }
                )

        for k in k_values:
            records[f"hit@{k}"].append(hit_at_k(ranked, target, k))
            records[f"precision@{k}"].append(precision_at_k(ranked, target, k))
            records[f"ndcg@{k}"].append(ndcg_at_k(ranked, target, k))
            records[f"mrr@{k}"].append(mrr_at_k(ranked, target, k))

    if (
        save_retrieval_traces
        and retrieval_trace_output_path is not None
        and retrieval_trace_rows
    ):
        retrieval_trace_output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(retrieval_trace_rows).to_csv(
            retrieval_trace_output_path,
            index=False,
        )

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
        or col.startswith("mrr@")
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
        or col.startswith("mrr@")
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
            "(1) semantic search with OpenAI embeddings and "
            "(2) BM25 keyword retrieval."
        )
    )
    parser.add_argument("--policy-input", type=Path, default=DEFAULT_POLICY_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--metric-top-k-values",
        "--k-values",
        dest="metric_top_k_values",
        type=str,
        default=DEFAULT_METRIC_TOP_K_VALUES,
        help=(
            "Top-k cutoffs for retrieval metrics (Hit@k/Precision@k/NDCG@k/MRR@k). "
            "This is not a time-series window setting."
        ),
    )
    parser.add_argument(
        "--openai-embedding-model",
        type=str,
        default=DEFAULT_OPENAI_EMBEDDING_MODEL,
        help="OpenAI embedding model for human-summary semantic retrieval.",
    )
    parser.add_argument(
        "--openai-embedding-batch-size",
        type=int,
        default=DEFAULT_OPENAI_EMBEDDING_BATCH_SIZE,
        help="Batch size for chunk embedding requests.",
    )
    parser.add_argument(
        "--chunk-vectordb-dir",
        type=Path,
        default=DEFAULT_CHUNK_VECTORDB_DIR,
        help="Persisted chunk vector DB directory built by scripts/3_build_chunk_vectordb.py.",
    )
    parser.add_argument(
        "--chunk-vectordb-collection",
        type=str,
        default=DEFAULT_CHUNK_VECTORDB_COLLECTION,
        help=(
            "Collection name for chunk vector DB. "
            "Use 'auto' or 'manifest' to resolve from vectordb manifest.json."
        ),
    )
    parser.add_argument(
        "--save-retrieval-traces",
        action="store_true",
        help=(
            "Save per-query retrieved chunk traces to retrieval_traces.csv under "
            "each fold/method directory."
        ),
    )
    parser.add_argument(
        "--retrieval-trace-top-k",
        type=int,
        default=0,
        help=(
            "Top chunks to save per query when --save-retrieval-traces is enabled. "
            "Use 0 to auto-resolve to max(k-values)."
        ),
    )
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument(
        "--significance-baseline", type=str, default="human_summary_chunk_semantic"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ensure_openai_api_key_loaded()

    k_values = parse_csv_ints(args.metric_top_k_values)
    if any(k <= 0 for k in k_values):
        raise ValueError("All k-values must be positive")
    if args.openai_embedding_batch_size <= 0:
        raise ValueError("--openai-embedding-batch-size must be > 0")
    if args.retrieval_trace_top_k < 0:
        raise ValueError("--retrieval-trace-top-k must be >= 0")

    policy_df = load_policy_df(args.policy_input)
    all_fold_ids = sorted(policy_df["fold"].unique().tolist())
    print(
        "[INFO] Loaded policy data: "
        f"rows={len(policy_df)}, folds={all_fold_ids}, k_values={k_values}"
    )
    print(
        "[INFO] This script does not use time-series windows; "
        "k refers to retrieval metric cutoffs only."
    )
    print(
        "[INFO] Retrieval config: "
        f"embedding_model={args.openai_embedding_model}, "
        f"chunk_vectordb_dir={args.chunk_vectordb_dir}, "
        f"chunk_collection={args.chunk_vectordb_collection}"
    )

    fold_ids = list(all_fold_ids)
    if args.fold is not None:
        if args.fold not in all_fold_ids:
            raise ValueError(f"Fold {args.fold} does not exist in {args.policy_input}")
        fold_ids = [args.fold]

    all_results: list[dict] = []
    show_progress = not args.no_progress

    for fold_id in fold_ids:
        chunk_fold_ids = [fold_id]

        test_df = policy_df[policy_df["fold"] == fold_id][
            ["Family Summary", "Document ID"]
        ].copy()
        test_df = test_df.drop_duplicates().reset_index(drop=True)

        chunk_corpus_df, chunk_embeddings = load_chunk_corpus_from_vectordb(
            vectordb_dir=args.chunk_vectordb_dir,
            collection_name=args.chunk_vectordb_collection,
            chunk_fold_ids=chunk_fold_ids,
        )

        print(
            "[INFO] Fold setup: "
            f"fold={fold_id}, queries={len(test_df)}, "
            f"chunks={len(chunk_corpus_df)}, "
            f"unique_docs={chunk_corpus_df['Document ID'].nunique()}"
        )

        chunk_texts = chunk_corpus_df["chunk_text"].astype(str).tolist()

        semantic = HumanSummaryChunkSemanticRetriever(
            chunk_texts=chunk_texts,
            model_name=args.openai_embedding_model,
            embedding_batch_size=args.openai_embedding_batch_size,
            precomputed_chunk_embeddings=chunk_embeddings,
        )
        bm25 = ChunkBM25Retriever(chunk_texts=chunk_texts)

        for method_name, retriever in [
            ("human_summary_chunk_semantic", semantic),
            ("chunk_bm25", bm25),
        ]:
            print(f"[INFO] Running method={method_name} on fold={fold_id}")
            run_dir = args.output_dir / f"fold_{fold_id}" / method_name
            result = evaluate_fold(
                fold_id=fold_id,
                method_name=method_name,
                retriever=retriever,
                test_df=test_df,
                chunk_corpus_df=chunk_corpus_df,
                k_values=k_values,
                save_retrieval_traces=args.save_retrieval_traces,
                retrieval_trace_top_k=args.retrieval_trace_top_k,
                retrieval_trace_output_path=(run_dir / "retrieval_traces.csv"),
                show_progress=show_progress,
            )
            result["fold"] = int(fold_id)
            all_results.append(result)

            print(
                f"Finished fold={fold_id}, method={method_name}, "
                f"hit@5={result.get('hit@5', 0.0):.4f}, "
                f"ndcg@5={result.get('ndcg@5', 0.0):.4f}, "
                f"precision@5={result.get('precision@5', 0.0):.4f}, "
                f"mrr@5={result.get('mrr@5', 0.0):.4f}"
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
        "task": "summary_to_chunk_retrieval_baselines",
        "folds": fold_ids,
        "chunk_folds_per_test_fold": {
            str(test_fold): [test_fold] for test_fold in fold_ids
        },
        "k_values": k_values,
        "metric_top_k_values": k_values,
        "openai_embedding_model": args.openai_embedding_model,
        "openai_embedding_batch_size": int(args.openai_embedding_batch_size),
        "policy_input": str(args.policy_input),
        "chunk_vectordb_dir": str(args.chunk_vectordb_dir),
        "chunk_vectordb_collection": args.chunk_vectordb_collection,
        "methods": ["human_summary_chunk_semantic", "chunk_bm25"],
        "metrics": ["Hit@k", "Precision@k", "NDCG@k", "MRR@k"],
        "results_csv": str(results_path),
        "significance_baseline": args.significance_baseline,
        "significance_csv": str(significance_path),
        "save_retrieval_traces": bool(args.save_retrieval_traces),
        "retrieval_trace_top_k": int(args.retrieval_trace_top_k),
        "progress_enabled": bool(show_progress),
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved results: {results_path}")
    print(f"Saved significance tests: {significance_path}")


if __name__ == "__main__":
    main()
