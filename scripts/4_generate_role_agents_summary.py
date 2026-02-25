import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import threading
import time
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
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

try:
    from dotenv import find_dotenv, load_dotenv
except Exception:  # pragma: no cover - optional dependency
    find_dotenv = None
    load_dotenv = None


DEFAULT_POLICY_INPUT = Path("data/csv/group_kfold_assignments.csv")
DEFAULT_MODEL_INPUT_DIR = Path("data/model_input/kfold")
DEFAULT_OUTPUT_DIR = Path("results/role_agent_summary_experiments")
DEFAULT_METRIC_TOP_K_VALUES = "1,5,10"
DEFAULT_WINDOWS = "1,2,5,10"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_EMBEDDING_BATCH_SIZE = 128
DEFAULT_OPENAI_MAX_INFLIGHT_REQUESTS = 6
DEFAULT_OPENAI_MAX_RETRIES = 5
DEFAULT_OPENAI_RETRY_BASE_SECONDS = 1.0
DEFAULT_OPENAI_RETRY_MAX_SECONDS = 20.0
DEFAULT_CHUNK_VECTORDB_DIR = Path("data/vectorstore/policy_chunks_chroma")
DEFAULT_CHUNK_VECTORDB_COLLECTION = "policy_chunks_openai"
DEFAULT_OWID_DIR = Path("data/owid")
DEFAULT_QUERY_COLUMN = "role_agent_summary"
DEFAULT_HUMAN_SUMMARY_COLUMN = "Family Summary"
DEFAULT_DOC_ID_COLUMN = "Document ID"
DEFAULT_FOLD_COLUMN = "fold"
FAST_RUN_FOLD_PARALLELISM = 2
FAST_RUN_ADVISOR_PARALLELISM = 3
FAST_RUN_RETRIEVAL_TRACE_TOP_K = 10
FAST_RUN_QUERY_PROGRESS_EVERY = 1

T = TypeVar("T")


def log_info(message: str) -> None:
    print(f"[INFO] {message}")


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


def parse_windows(raw: str) -> list[int]:
    windows = parse_csv_ints(raw)
    if any(window <= 0 for window in windows):
        raise ValueError("All window values must be positive integers")
    return windows


def discover_table_key_domain_map(owid_dir: Path) -> dict[str, str]:
    if not owid_dir.exists():
        return {}

    mapping: dict[str, str] = {}
    meta_paths = sorted(owid_dir.rglob("*.meta.json"))
    for idx, meta_path in enumerate(meta_paths):
        table_key = f"t{idx:03d}"
        try:
            rel = meta_path.relative_to(owid_dir)
            parts = rel.parts
        except Exception:
            parts = ()
        domain = parts[0] if parts else ""
        mapping[table_key] = domain
    return mapping


def load_fold_feature_columns(model_input_dir: Path, fold_id: int) -> list[str]:
    fold_dir = model_input_dir / f"fold_{fold_id}"
    test_csv_path = fold_dir / "scaled_test_time_series.csv"
    if not test_csv_path.exists():
        return []

    frame = pd.read_csv(test_csv_path, nrows=0)
    return [col for col in frame.columns.tolist() if col not in {"country", "year"}]


def build_domain_feature_indices(
    feature_columns: list[str],
    table_key_domain_map: dict[str, str],
) -> dict[str, list[int]]:
    domain_to_indices = {
        "climate_data": [],
        "socio_economics_data": [],
        "other_data": [],
    }

    for index, column in enumerate(feature_columns):
        if "__" not in column:
            continue
        table_key = column.split("__", 1)[0]
        domain = table_key_domain_map.get(table_key, "")
        if domain in domain_to_indices:
            domain_to_indices[domain].append(index)

    return domain_to_indices


def build_domain_feature_names(
    feature_columns: list[str],
    domain_feature_indices: dict[str, list[int]],
) -> dict[str, list[str]]:
    domain_feature_names: dict[str, list[str]] = {}
    for domain, indices in domain_feature_indices.items():
        names: list[str] = []
        for idx in indices:
            if 0 <= idx < len(feature_columns):
                names.append(feature_columns[idx])
        domain_feature_names[domain] = names
    return domain_feature_names


def load_policy_df(
    path: Path,
    fold_col: str,
    doc_id_col: str,
    summary_col: str,
    query_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {fold_col, doc_id_col, summary_col}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    out = df.copy()
    out[summary_col] = out[summary_col].apply(clean_text)
    out[doc_id_col] = out[doc_id_col].astype(str)
    out[fold_col] = out[fold_col].astype(int)

    if query_col not in out.columns:
        out[query_col] = ""
    out[query_col] = out[query_col].apply(clean_text)

    out = out[(out[summary_col] != "") & (out[doc_id_col] != "")]
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


class OpenAIRequestController:
    def __init__(
        self,
        *,
        max_inflight_requests: int,
        max_retries: int,
        retry_base_seconds: float,
        retry_max_seconds: float,
    ):
        if max_inflight_requests <= 0:
            raise ValueError("max_inflight_requests must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if retry_base_seconds <= 0:
            raise ValueError("retry_base_seconds must be > 0")
        if retry_max_seconds <= 0:
            raise ValueError("retry_max_seconds must be > 0")

        self._semaphore = threading.BoundedSemaphore(max_inflight_requests)
        self.max_retries = int(max_retries)
        self.retry_base_seconds = float(retry_base_seconds)
        self.retry_max_seconds = float(retry_max_seconds)

    def chat_completion(self, client: object, **kwargs) -> object:
        attempt = 0
        while True:
            with self._semaphore:
                try:
                    return client.chat.completions.create(**kwargs)
                except Exception as exc:
                    if attempt >= self.max_retries:
                        raise RuntimeError(
                            f"OpenAI request failed after {attempt + 1} attempts"
                        ) from exc

            sleep_seconds = min(
                self.retry_max_seconds,
                self.retry_base_seconds * (2**attempt),
            )
            jitter = random.uniform(0.0, min(1.0, sleep_seconds * 0.25))
            time.sleep(sleep_seconds + jitter)
            attempt += 1


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


class RoleSummaryChunkSemanticRetriever:
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
            log_info(
                "Using precomputed chunk embeddings "
                f"with shape={tuple(precomputed_chunk_embeddings.shape)}"
            )
            if precomputed_chunk_embeddings.shape[0] != len(chunk_texts):
                raise ValueError(
                    "precomputed_chunk_embeddings row count must match chunk_texts length"
                )
            self.chunk_embeddings = np.asarray(
                precomputed_chunk_embeddings, dtype=np.float64
            )
        else:
            log_info(f"Computing chunk embeddings for {len(chunk_texts)} chunks")
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
    window: int,
    method_name: str,
    retriever: object,
    test_df: pd.DataFrame,
    query_col: str,
    chunk_corpus_df: pd.DataFrame,
    k_values: list[int],
    save_retrieval_traces: bool,
    retrieval_trace_top_k: int,
    retrieval_trace_output_path: Optional[Path],
    show_progress: bool,
    log_query_progress: bool,
    query_progress_every: int,
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
    queries_seen = 0
    query_iter: Iterable[tuple[int, pd.Series]] = test_df.iterrows()
    if show_progress:
        query_iter = with_progress(
            query_iter,
            total=len(test_df),
            desc=f"fold {fold_id} | window {window} | {method_name} queries",
            unit="query",
        )

    for query_index, row in query_iter:
        queries_seen += 1
        query = clean_text(row.get(query_col, ""))
        target = str(row.get("Document ID", ""))
        if not query or not target:
            skipped += 1
            if log_query_progress and queries_seen % query_progress_every == 0:
                log_info(
                    f"Retrieval progress | fold={fold_id} window={window} method={method_name} "
                    f"query={queries_seen}/{len(test_df)} status=skipped"
                )
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
                        "fold": int(fold_id),
                        "window": int(window),
                        "query_index": int(query_index),
                        "query_text": query,
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

        if log_query_progress and queries_seen % query_progress_every == 0:
            log_info(
                f"Retrieval progress | fold={fold_id} window={window} method={method_name} "
                f"query={queries_seen}/{len(test_df)}"
            )

    if (
        log_query_progress
        and (len(test_df) > 0)
        and (queries_seen % query_progress_every != 0)
    ):
        log_info(
            f"Retrieval progress | fold={fold_id} window={window} method={method_name} "
            f"query={queries_seen}/{len(test_df)}"
        )

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

    metrics["fold"] = int(fold_id)
    metrics["window"] = int(window)
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

    group_cols = ["method"]
    if "window" in results_df.columns:
        group_cols = ["window", "method"]

    grouped = (
        results_df.groupby(group_cols, as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values(group_cols)
    )
    grouped.to_csv(output_dir / "summary_mean_metrics.csv", index=False)


def load_window_test_records(
    model_input_dir: Path, fold_id: int, window: int
) -> list[dict]:
    path = model_input_dir / f"fold_{fold_id}" / f"window_{window}" / "test.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing fold/window test file: {path}. "
            "Run scripts/2_data_preparation_and_time_series_scaling.py first."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    return payload


def transform_time_series_to_text(
    values: object,
    *,
    table_title: str = "time_series_window",
    selected_column_indices: Optional[list[int]] = None,
    selected_column_names: Optional[list[str]] = None,
    max_rows: int = 10,
    max_cols: int = 22,
) -> str:
    try:
        matrix = np.asarray(values, dtype=float)
    except Exception:
        return ""

    if matrix.ndim != 2:
        return ""

    original_col_count = matrix.shape[1]

    if selected_column_indices is not None:
        if len(selected_column_indices) == 0:
            return ""
        valid_indices = [
            idx
            for idx in selected_column_indices
            if isinstance(idx, int) and 0 <= idx < matrix.shape[1]
        ]
        if not valid_indices:
            return ""
        matrix = matrix[:, valid_indices]
        if selected_column_names is not None:
            aligned_names: list[str] = []
            for position, original_idx in enumerate(selected_column_indices):
                if (
                    isinstance(original_idx, int)
                    and 0 <= original_idx < original_col_count
                    and position < len(selected_column_names)
                ):
                    aligned_names.append(str(selected_column_names[position]))
            selected_column_names = aligned_names

    rows = min(matrix.shape[0], max_rows)
    cols = min(matrix.shape[1], max_cols)
    sliced = matrix[:rows, :cols]
    column_names: list[str] = []
    if selected_column_names is not None:
        column_names = [str(name) for name in selected_column_names[:cols]]

    table_rows: list[str] = []
    for row in sliced:
        if len(column_names) == len(row):
            formatted_cells = [
                f"{name}={float(value):.2f}" for name, value in zip(column_names, row)
            ]
        else:
            formatted_cells = [f"{float(value):.2f}" for value in row]
        table_rows.append(",".join(formatted_cells))

    table_csv = "\n".join(table_rows).strip()
    if not table_csv:
        return ""
    prefixed_rows = "\n".join(f"[ROW] {line}" for line in table_csv.splitlines())
    return (
        f"[START_OF_TABLE]\\n"
        f"{table_title}\\n"
        f"{prefixed_rows}\\n"
        f"[END_OF_TABLE]\\n"
    )


def role_agent_chain_generate_summary(
    client: object,
    openai_controller: OpenAIRequestController,
    model_name: str,
    climate_ts_table: str,
    socio_economic_ts_table: str,
    other_ts_table: str,
    country_context: str = "",
    sector_context: str = "",
    window_context: Optional[int] = None,
    advisor_parallelism: int = 3,
) -> str:
    if not (climate_ts_table or socio_economic_ts_table or other_ts_table):
        return ""

    context_line = (
        f"Country: {country_context or 'unknown'} | "
        f"Sector: {sector_context or 'unknown'} | "
        f"History window: {window_context if window_context is not None else 'unknown'}"
    )

    max_workers = max(1, min(int(advisor_parallelism), 3))

    def _run_advisor_call(system_prompt: str, user_prompt: str) -> str:
        completion = openai_controller.chat_completion(
            client,
            model=model_name,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return clean_text(completion.choices[0].message.content)

    advisor_prompts: list[tuple[str, str, str]] = [
        (
            "climate",
            "You are a climate-data advisor. Return concise insights from the time-series table.",
            (
                "Analyze climate signals from this normalized time-series table.\n"
                f"{context_line}\n"
                f"Table:\n{climate_ts_table}"
            ),
        ),
        (
            "socioeconomic",
            "You are a socio-economic advisor. Return concise implications from the time-series table.",
            (
                "Analyze socio-economic signals from this normalized time-series table.\n"
                f"{context_line}\n"
                f"Table:\n{socio_economic_ts_table}"
            ),
        ),
        (
            "other",
            "You are an infrastructure-and-context advisor. Return concise implications from the time-series table.",
            (
                "Analyze contextual signals from this normalized time-series table.\n"
                f"{context_line}\n"
                f"Table:\n{other_ts_table}"
            ),
        ),
    ]

    advisor_outputs: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            name: executor.submit(_run_advisor_call, system_prompt, user_prompt)
            for name, system_prompt, user_prompt in advisor_prompts
        }
        for name, future in futures.items():
            advisor_outputs[name] = future.result()

    climate_text = advisor_outputs.get("climate", "")
    socioeconomic_text = advisor_outputs.get("socioeconomic", "")
    other_text = advisor_outputs.get("other", "")

    policy_expert = openai_controller.chat_completion(
        client,
        model=model_name,
        temperature=0.25,
        messages=[
            {
                "role": "system",
                "content": "You are a policy expert. Synthesize advisor outputs into an actionable policy brief.",
            },
            {
                "role": "user",
                "content": (
                    f"{context_line}\n\n"
                    f"Climate advisor:\n{climate_text}\n\n"
                    f"Socio-economic advisor:\n{socioeconomic_text}\n\n"
                    f"Other-data advisor:\n{other_text}"
                ),
            },
        ],
    )
    policy_text = clean_text(policy_expert.choices[0].message.content)

    retrieval_summary_draft = openai_controller.chat_completion(
        client,
        model=model_name,
        temperature=0.4,
        messages=[
            {
                "role": "system",
                "content": "Summarize the policy expert output into one concise retrieval-friendly paragraph.",
            },
            {"role": "user", "content": policy_text},
        ],
    )
    retrieval_summary_draft_text = clean_text(
        retrieval_summary_draft.choices[0].message.content
    )

    retrieval_summary_final = openai_controller.chat_completion(
        client,
        model=model_name,
        temperature=0.4,
        messages=[
            {
                "role": "system",
                "content": "Rewrite the summary for semantic retrieval: keep entities, sectors, and targets explicit.",
            },
            {"role": "user", "content": retrieval_summary_draft_text},
        ],
    )
    retrieval_summary_final_text = clean_text(
        retrieval_summary_final.choices[0].message.content
    )

    retrieval_summary_compact = openai_controller.chat_completion(
        client,
        model=model_name,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Convert the policy summary into a compact retrieval query. "
                    "Keep country, sectors, policy instruments, numeric targets, and years explicit. "
                    "Return one dense paragraph under 90 words."
                ),
            },
            {"role": "user", "content": retrieval_summary_final_text},
        ],
    )
    return clean_text(retrieval_summary_compact.choices[0].message.content)


def build_window_test_dataframe(
    policy_df: pd.DataFrame,
    fold_id: int,
    window: int,
    window_records: list[dict],
    fold_col: str,
    doc_id_col: str,
    summary_col: str,
    query_col: str,
    generate_role_summary_from_time_series: bool,
    openai_model: str,
    openai_controller: OpenAIRequestController,
    generated_cache_path: Optional[Path],
    domain_feature_indices: dict[str, list[int]],
    domain_feature_names: dict[str, list[str]],
    advisor_parallelism: int,
    show_progress: bool,
    log_query_progress: bool,
    query_progress_every: int,
) -> pd.DataFrame:
    fold_policy = policy_df[policy_df[fold_col] == fold_id].copy()
    if fold_policy.empty:
        return pd.DataFrame(columns=["Document ID", "Family Summary", query_col])

    doc_to_summary: dict[str, str] = {}
    doc_to_role_query: dict[str, str] = {}
    for _, row in fold_policy.iterrows():
        doc_id = str(row.get(doc_id_col, "")).strip()
        if not doc_id:
            continue
        doc_to_summary[doc_id] = clean_text(row.get(summary_col, ""))
        existing_query = clean_text(row.get(query_col, ""))
        if existing_query and doc_id not in doc_to_role_query:
            doc_to_role_query[doc_id] = existing_query

    cached_generated: dict[str, str] = {}
    if generated_cache_path is not None and generated_cache_path.exists():
        try:
            cache_df = pd.read_csv(generated_cache_path)
            for _, row in cache_df.iterrows():
                cache_key = str(row.get("cache_key", "")).strip()
                value = clean_text(row.get(query_col, ""))
                if cache_key and value:
                    cached_generated[cache_key] = value
        except Exception:
            cached_generated = {}

    rows: list[dict[str, str]] = []
    generated_rows: list[dict[str, str]] = []

    client = None
    if generate_role_summary_from_time_series:
        ensure_openai_api_key_loaded()
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "Missing dependency `openai`. Install with `pip install openai`."
            ) from exc
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip())

    iterator: Iterable[tuple[int, dict]] = enumerate(window_records)
    if show_progress:
        iterator = with_progress(
            iterator,
            total=len(window_records),
            desc=f"fold {fold_id} | window {window} build test",
            unit="row",
        )

    for index, record in iterator:
        row_number = index + 1
        target_doc_id = str(record.get("doc_id", "")).strip()
        if not target_doc_id:
            if log_query_progress and row_number % query_progress_every == 0:
                log_info(
                    f"Build progress | fold={fold_id} window={window} "
                    f"row={row_number}/{len(window_records)} status=skipped"
                )
            continue

        family_summary = doc_to_summary.get(target_doc_id, "")

        role_query = doc_to_role_query.get(target_doc_id, "")
        if generate_role_summary_from_time_series:
            country_context = clean_text(record.get("country", ""))
            raw_sector = record.get("sector", "")
            if isinstance(raw_sector, list):
                sector_context = ", ".join(
                    str(value) for value in raw_sector if str(value)
                )
            else:
                sector_context = clean_text(raw_sector)

            climate_ts_table = transform_time_series_to_text(
                record.get("positive_time_series", []),
                table_title=f"fold_{fold_id}_window_{window}_record_{index}_climate_data",
                selected_column_indices=domain_feature_indices.get("climate_data", []),
                selected_column_names=domain_feature_names.get("climate_data", []),
            )
            socio_ts_table = transform_time_series_to_text(
                record.get("positive_time_series", []),
                table_title=f"fold_{fold_id}_window_{window}_record_{index}_socio_economics_data",
                selected_column_indices=domain_feature_indices.get(
                    "socio_economics_data", []
                ),
                selected_column_names=domain_feature_names.get(
                    "socio_economics_data", []
                ),
            )
            other_ts_table = transform_time_series_to_text(
                record.get("positive_time_series", []),
                table_title=f"fold_{fold_id}_window_{window}_record_{index}_other_data",
                selected_column_indices=domain_feature_indices.get("other_data", []),
                selected_column_names=domain_feature_names.get("other_data", []),
            )
            cache_key_src = (
                f"fold={fold_id}|window={window}|idx={index}|doc_id={target_doc_id}"
            )
            cache_key = hashlib.sha256(cache_key_src.encode("utf-8")).hexdigest()

            if cache_key in cached_generated:
                role_query = cached_generated[cache_key]
                if log_query_progress and row_number % query_progress_every == 0:
                    log_info(
                        f"Build progress | fold={fold_id} window={window} "
                        f"row={row_number}/{len(window_records)} status=cache_hit"
                    )
            else:
                role_query = role_agent_chain_generate_summary(
                    client=client,
                    openai_controller=openai_controller,
                    model_name=openai_model,
                    climate_ts_table=climate_ts_table,
                    socio_economic_ts_table=socio_ts_table,
                    other_ts_table=other_ts_table,
                    country_context=country_context,
                    sector_context=sector_context,
                    window_context=window,
                    advisor_parallelism=advisor_parallelism,
                )
                if log_query_progress and row_number % query_progress_every == 0:
                    log_info(
                        f"Build progress | fold={fold_id} window={window} "
                        f"row={row_number}/{len(window_records)} status=generated"
                    )
                if role_query:
                    cached_generated[cache_key] = role_query

            if role_query:
                generated_rows.append(
                    {
                        "fold": str(fold_id),
                        "window": str(window),
                        "record_index": str(index),
                        "cache_key": cache_key,
                        query_col: role_query,
                        "target_doc_id": target_doc_id,
                    }
                )

        rows.append(
            {
                "Document ID": target_doc_id,
                "Family Summary": family_summary,
                query_col: role_query,
            }
        )

        if (
            log_query_progress
            and not generate_role_summary_from_time_series
            and row_number % query_progress_every == 0
        ):
            log_info(
                f"Build progress | fold={fold_id} window={window} "
                f"row={row_number}/{len(window_records)}"
            )

    if (
        log_query_progress
        and len(window_records) > 0
        and len(window_records) % query_progress_every != 0
    ):
        log_info(
            f"Build progress | fold={fold_id} window={window} "
            f"row={len(window_records)}/{len(window_records)}"
        )

    if generated_cache_path is not None and generated_rows:
        generated_cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(generated_rows).drop_duplicates().to_csv(
            generated_cache_path, index=False
        )

    if not rows:
        return pd.DataFrame(columns=["Document ID", "Family Summary", query_col])

    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run role-agent summary retrieval over fold/window test sets with chunk-level "
            "OpenAI semantic retrieval."
        )
    )

    parser.add_argument("--policy-input", type=Path, default=DEFAULT_POLICY_INPUT)
    parser.add_argument("--model-input-dir", type=Path, default=DEFAULT_MODEL_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--metric-top-k-values",
        "--k-values",
        dest="metric_top_k_values",
        type=str,
        default=DEFAULT_METRIC_TOP_K_VALUES,
        help="Top-k cutoffs for retrieval metrics (Hit@k/Precision@k/NDCG@k/MRR@k).",
    )
    parser.add_argument("--windows", type=str, default=DEFAULT_WINDOWS)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument(
        "--fast-run",
        action="store_true",
        help=(
            "Convenience mode: enables role-summary generation, fold/advisor parallelism, "
            "retrieval traces, and per-query progress with recommended defaults."
        ),
    )
    parser.add_argument(
        "--fold-parallelism",
        type=int,
        default=1,
        help="Number of folds to process concurrently. 1 disables fold-level parallelism.",
    )

    parser.add_argument("--fold-column", type=str, default=DEFAULT_FOLD_COLUMN)
    parser.add_argument("--doc-id-column", type=str, default=DEFAULT_DOC_ID_COLUMN)
    parser.add_argument(
        "--human-summary-column", type=str, default=DEFAULT_HUMAN_SUMMARY_COLUMN
    )
    parser.add_argument("--query-column", type=str, default=DEFAULT_QUERY_COLUMN)

    parser.add_argument(
        "--owid-dir",
        type=Path,
        default=DEFAULT_OWID_DIR,
        help="OWID root directory used to infer table domains (climate/socio/other).",
    )

    parser.add_argument(
        "--openai-embedding-model",
        type=str,
        default=DEFAULT_OPENAI_EMBEDDING_MODEL,
        help="OpenAI embedding model for chunk semantic retrieval.",
    )
    parser.add_argument(
        "--openai-embedding-batch-size",
        type=int,
        default=DEFAULT_OPENAI_EMBEDDING_BATCH_SIZE,
        help="Batch size for chunk embedding requests.",
    )
    parser.add_argument(
        "--openai-max-inflight-requests",
        type=int,
        default=DEFAULT_OPENAI_MAX_INFLIGHT_REQUESTS,
        help="Global max concurrent OpenAI chat requests across threads.",
    )
    parser.add_argument(
        "--openai-max-retries",
        type=int,
        default=DEFAULT_OPENAI_MAX_RETRIES,
        help="Max retries for transient OpenAI chat failures.",
    )
    parser.add_argument(
        "--openai-retry-base-seconds",
        type=float,
        default=DEFAULT_OPENAI_RETRY_BASE_SECONDS,
        help="Initial backoff seconds for OpenAI retries.",
    )
    parser.add_argument(
        "--openai-retry-max-seconds",
        type=float,
        default=DEFAULT_OPENAI_RETRY_MAX_SECONDS,
        help="Maximum backoff seconds for OpenAI retries.",
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
        "--generate-role-summary-from-time-series",
        action="store_true",
        help="Generate role-agent summaries from fold/window test time-series records via OpenAI chain.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="Model used for role summary generation when enabled.",
    )
    parser.add_argument(
        "--advisor-parallelism",
        type=int,
        default=3,
        help="Max concurrent advisor API calls per generated query (1-3).",
    )
    parser.add_argument(
        "--save-generated-role-summaries",
        action="store_true",
        help="Save generated role summaries for each fold/window.",
    )

    parser.add_argument(
        "--save-retrieval-traces",
        action="store_true",
        help="Save per-query retrieved chunk traces under each fold/window/method directory.",
    )
    parser.add_argument(
        "--retrieval-trace-top-k",
        type=int,
        default=0,
        help="Top chunks to save per query when --save-retrieval-traces is enabled. 0 => max(k-values).",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--log-query-progress",
        action="store_true",
        help="Log retriever progress by query during evaluation.",
    )
    parser.add_argument(
        "--query-progress-every",
        type=int,
        default=1,
        help="Emit query-progress log every N queries when --log-query-progress is enabled.",
    )

    return parser.parse_args()


def process_fold(
    fold_id: int,
    *,
    args: argparse.Namespace,
    policy_df: pd.DataFrame,
    windows: list[int],
    k_values: list[int],
    show_progress: bool,
    table_key_domain_map: dict[str, str],
    openai_controller: OpenAIRequestController,
) -> list[dict]:
    fold_results: list[dict] = []

    log_info(f"Preparing fold {fold_id}")
    chunk_fold_ids = [fold_id]
    chunk_corpus_df, chunk_embeddings = load_chunk_corpus_from_vectordb(
        vectordb_dir=args.chunk_vectordb_dir,
        collection_name=args.chunk_vectordb_collection,
        chunk_fold_ids=chunk_fold_ids,
    )
    log_info(f"Fold {fold_id}: loaded {len(chunk_corpus_df)} chunk rows")
    log_info(
        f"Fold {fold_id}: using {'stored' if chunk_embeddings is not None else 'freshly computed'} chunk embeddings"
    )

    chunk_texts = chunk_corpus_df["chunk_text"].astype(str).tolist()
    role_semantic = RoleSummaryChunkSemanticRetriever(
        chunk_texts=chunk_texts,
        model_name=args.openai_embedding_model,
        embedding_batch_size=args.openai_embedding_batch_size,
        precomputed_chunk_embeddings=chunk_embeddings,
    )
    feature_columns = load_fold_feature_columns(args.model_input_dir, fold_id)
    domain_feature_indices = build_domain_feature_indices(
        feature_columns=feature_columns,
        table_key_domain_map=table_key_domain_map,
    )
    domain_feature_names = build_domain_feature_names(
        feature_columns=feature_columns,
        domain_feature_indices=domain_feature_indices,
    )
    log_info(
        f"Fold {fold_id}: feature columns={len(feature_columns)} "
        f"(climate={len(domain_feature_indices.get('climate_data', []))}, "
        f"socio={len(domain_feature_indices.get('socio_economics_data', []))}, "
        f"other={len(domain_feature_indices.get('other_data', []))})"
    )

    for window in windows:
        log_info(f"Fold {fold_id} | Window {window}: loading test records")
        window_records = load_window_test_records(
            model_input_dir=args.model_input_dir,
            fold_id=fold_id,
            window=window,
        )
        log_info(
            f"Fold {fold_id} | Window {window}: loaded {len(window_records)} raw test rows"
        )

        generated_cache_path = None
        if args.save_generated_role_summaries:
            generated_cache_path = (
                args.output_dir
                / f"fold_{fold_id}"
                / f"window_{window}"
                / "generated_role_agent_summaries.csv"
            )

        test_df = build_window_test_dataframe(
            policy_df=policy_df,
            fold_id=fold_id,
            window=window,
            window_records=window_records,
            fold_col=args.fold_column,
            doc_id_col=args.doc_id_column,
            summary_col=args.human_summary_column,
            query_col=args.query_column,
            generate_role_summary_from_time_series=args.generate_role_summary_from_time_series,
            openai_model=args.openai_model,
            openai_controller=openai_controller,
            generated_cache_path=generated_cache_path,
            domain_feature_indices=domain_feature_indices,
            domain_feature_names=domain_feature_names,
            advisor_parallelism=args.advisor_parallelism,
            show_progress=show_progress,
            log_query_progress=args.log_query_progress,
            query_progress_every=args.query_progress_every,
        )
        log_info(
            f"Fold {fold_id} | Window {window}: built {len(test_df)} evaluation rows"
        )

        non_empty_query_count = int(
            test_df[args.query_column].astype(str).str.strip().ne("").sum()
        )
        log_info(
            f"Fold {fold_id} | Window {window}: non-empty queries={non_empty_query_count}/{len(test_df)}"
        )
        if non_empty_query_count == 0:
            raise RuntimeError(
                f"Fold {fold_id}, window {window}: all queries in column `{args.query_column}` are empty. "
                "Use a populated query column or enable `--generate-role-summary-from-time-series`."
            )

        if test_df.empty:
            print(f"[WARN] Skip fold={fold_id}, window={window}: empty test dataframe")
            continue

        method_specs: list[tuple[str, object, str]] = [
            (
                "role_agent_summary_chunk_semantic",
                role_semantic,
                args.query_column,
            )
        ]

        for method_name, retriever, query_col in method_specs:
            run_dir = (
                args.output_dir / f"fold_{fold_id}" / f"window_{window}" / method_name
            )
            result = evaluate_fold(
                fold_id=fold_id,
                window=window,
                method_name=method_name,
                retriever=retriever,
                test_df=test_df,
                query_col=query_col,
                chunk_corpus_df=chunk_corpus_df,
                k_values=k_values,
                save_retrieval_traces=args.save_retrieval_traces,
                retrieval_trace_top_k=args.retrieval_trace_top_k,
                retrieval_trace_output_path=(run_dir / "retrieval_traces.csv"),
                show_progress=show_progress,
                log_query_progress=args.log_query_progress,
                query_progress_every=args.query_progress_every,
            )
            fold_results.append(result)

            print(
                f"Finished fold={fold_id}, window={window}, method={method_name}, "
                f"hit@5={result.get('hit@5', 0.0):.4f}, "
                f"ndcg@5={result.get('ndcg@5', 0.0):.4f}, "
                f"precision@5={result.get('precision@5', 0.0):.4f}, "
                f"mrr@5={result.get('mrr@5', 0.0):.4f}"
            )

    return fold_results


def main() -> None:
    args = parse_args()

    if args.fast_run:
        args.generate_role_summary_from_time_series = True
        if args.fold_parallelism == 1:
            args.fold_parallelism = FAST_RUN_FOLD_PARALLELISM
        if args.advisor_parallelism == 3:
            args.advisor_parallelism = FAST_RUN_ADVISOR_PARALLELISM
        if not args.save_retrieval_traces:
            args.save_retrieval_traces = True
        if args.retrieval_trace_top_k == 0:
            args.retrieval_trace_top_k = FAST_RUN_RETRIEVAL_TRACE_TOP_K
        if not args.log_query_progress:
            args.log_query_progress = True
        if args.query_progress_every == 1:
            args.query_progress_every = FAST_RUN_QUERY_PROGRESS_EVERY

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_openai_api_key_loaded()
    log_info("Starting role-agent summary retrieval run")

    k_values = parse_csv_ints(args.metric_top_k_values)
    if any(k <= 0 for k in k_values):
        raise ValueError("All k-values must be positive")
    windows = parse_windows(args.windows)
    if args.window is not None:
        if args.window <= 0:
            raise ValueError("--window must be positive")
        windows = [args.window]
    if args.openai_embedding_batch_size <= 0:
        raise ValueError("--openai-embedding-batch-size must be > 0")
    if args.openai_max_inflight_requests <= 0:
        raise ValueError("--openai-max-inflight-requests must be > 0")
    if args.openai_max_retries < 0:
        raise ValueError("--openai-max-retries must be >= 0")
    if args.openai_retry_base_seconds <= 0:
        raise ValueError("--openai-retry-base-seconds must be > 0")
    if args.openai_retry_max_seconds <= 0:
        raise ValueError("--openai-retry-max-seconds must be > 0")
    if args.openai_retry_max_seconds < args.openai_retry_base_seconds:
        raise ValueError(
            "--openai-retry-max-seconds must be >= --openai-retry-base-seconds"
        )
    if args.fold_parallelism <= 0:
        raise ValueError("--fold-parallelism must be > 0")
    if args.advisor_parallelism <= 0:
        raise ValueError("--advisor-parallelism must be > 0")
    if args.retrieval_trace_top_k < 0:
        raise ValueError("--retrieval-trace-top-k must be >= 0")
    if args.query_progress_every <= 0:
        raise ValueError("--query-progress-every must be > 0")

    policy_df = load_policy_df(
        path=args.policy_input,
        fold_col=args.fold_column,
        doc_id_col=args.doc_id_column,
        summary_col=args.human_summary_column,
        query_col=args.query_column,
    )

    fold_ids = sorted(policy_df[args.fold_column].unique().tolist())
    if args.fold is not None:
        if args.fold not in fold_ids:
            raise ValueError(
                f"Fold {args.fold} does not exist in {args.policy_input} for column `{args.fold_column}`"
            )
        fold_ids = [args.fold]

    show_progress = not args.no_progress
    effective_show_progress = show_progress
    if args.fold_parallelism > 1 and show_progress:
        log_info(
            "Disabling tqdm progress bars because fold-level parallelism is enabled"
        )
        effective_show_progress = False
    table_key_domain_map = discover_table_key_domain_map(args.owid_dir)
    log_info(
        "Configuration: "
        f"fold={args.fold if args.fold is not None else 'all'}, "
        f"windows={windows}, "
        f"k_values={k_values}, "
        f"query_column={args.query_column}, "
        f"fast_run={args.fast_run}, "
        f"fold_parallelism={args.fold_parallelism}, "
        f"generate_role_summary={args.generate_role_summary_from_time_series}, "
        f"advisor_parallelism={args.advisor_parallelism}, "
        f"openai_max_inflight_requests={args.openai_max_inflight_requests}, "
        f"openai_max_retries={args.openai_max_retries}, "
        f"log_query_progress={args.log_query_progress}, "
        f"query_progress_every={args.query_progress_every}, "
        f"save_retrieval_traces={args.save_retrieval_traces}"
    )

    openai_controller = OpenAIRequestController(
        max_inflight_requests=args.openai_max_inflight_requests,
        max_retries=args.openai_max_retries,
        retry_base_seconds=args.openai_retry_base_seconds,
        retry_max_seconds=args.openai_retry_max_seconds,
    )
    all_results: list[dict] = []

    max_fold_workers = min(args.fold_parallelism, len(fold_ids))
    if max_fold_workers <= 1:
        for fold_id in fold_ids:
            fold_results = process_fold(
                fold_id,
                args=args,
                policy_df=policy_df,
                windows=windows,
                k_values=k_values,
                show_progress=effective_show_progress,
                table_key_domain_map=table_key_domain_map,
                openai_controller=openai_controller,
            )
            all_results.extend(fold_results)
    else:
        log_info(f"Running folds in parallel with workers={max_fold_workers}")
        with ThreadPoolExecutor(max_workers=max_fold_workers) as executor:
            futures = {
                executor.submit(
                    process_fold,
                    fold_id,
                    args=args,
                    policy_df=policy_df,
                    windows=windows,
                    k_values=k_values,
                    show_progress=effective_show_progress,
                    table_key_domain_map=table_key_domain_map,
                    openai_controller=openai_controller,
                ): fold_id
                for fold_id in fold_ids
            }
            for future in as_completed(futures):
                fold_id = futures[future]
                fold_results = future.result()
                all_results.extend(fold_results)
                log_info(
                    f"Fold {fold_id} completed in parallel with {len(fold_results)} result rows"
                )

    if not all_results:
        raise RuntimeError("No fold/window evaluation results were produced.")

    results_df = pd.DataFrame(all_results).sort_values(["fold", "window", "method"])
    results_path = args.output_dir / "all_fold_results.csv"
    results_df.to_csv(results_path, index=False)
    log_info(f"Saved fold results: {results_path}")

    summarize_fold_metrics(results_df, args.output_dir)
    log_info(f"Saved mean metrics: {args.output_dir / 'summary_mean_metrics.csv'}")

    metadata = {
        "task": "role_agent_summary_retrieval_experiments",
        "policy_input": str(args.policy_input),
        "model_input_dir": str(args.model_input_dir),
        "fold_column": args.fold_column,
        "doc_id_column": args.doc_id_column,
        "human_summary_column": args.human_summary_column,
        "query_column": args.query_column,
        "fast_run": bool(args.fast_run),
        "generated_role_summary_from_time_series": bool(
            args.generate_role_summary_from_time_series
        ),
        "fold_parallelism": int(args.fold_parallelism),
        "openai_model": args.openai_model,
        "advisor_parallelism": int(args.advisor_parallelism),
        "folds": fold_ids,
        "windows": windows,
        "k_values": k_values,
        "metric_top_k_values": k_values,
        "owid_dir": str(args.owid_dir),
        "openai_embedding_model": args.openai_embedding_model,
        "openai_embedding_batch_size": int(args.openai_embedding_batch_size),
        "openai_max_inflight_requests": int(args.openai_max_inflight_requests),
        "openai_max_retries": int(args.openai_max_retries),
        "openai_retry_base_seconds": float(args.openai_retry_base_seconds),
        "openai_retry_max_seconds": float(args.openai_retry_max_seconds),
        "chunk_vectordb_dir": str(args.chunk_vectordb_dir),
        "chunk_vectordb_collection": args.chunk_vectordb_collection,
        "methods": ["role_agent_summary_chunk_semantic"],
        "metrics": ["Hit@k", "Precision@k", "NDCG@k", "MRR@k"],
        "results_csv": str(results_path),
        "save_retrieval_traces": bool(args.save_retrieval_traces),
        "retrieval_trace_top_k": int(args.retrieval_trace_top_k),
        "log_query_progress": bool(args.log_query_progress),
        "query_progress_every": int(args.query_progress_every),
        "progress_enabled": bool(effective_show_progress),
    }

    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    log_info(f"Saved run metadata: {args.output_dir / 'run_metadata.json'}")

    print(f"Saved results: {results_path}")


if __name__ == "__main__":
    main()
