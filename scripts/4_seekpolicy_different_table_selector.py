import argparse
import concurrent.futures
import json
import os
import threading
import time
from ast import literal_eval
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

try:
    from langchain_openai import OpenAIEmbeddings
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Missing dependency `langchain-openai`. Install with `pip install langchain-openai`."
    ) from exc

# --- Table selection: embedding-based from table_selector ---
from table_selector import (
    discover_owid_titles,
    OpenAIEmbeddings,
    embed_texts,
    normalize_rows,
    select_titles_for_policies,
    configure_openai_concurrency,
    ensure_openai_api_key_loaded,
)

DEFAULT_POLICY_INPUT = Path("data/csv/group_kfold_assignments.csv")
DEFAULT_MODEL_INPUT_DIR = Path("data/model_input/kfold")
DEFAULT_OUTPUT_DIR = Path("results/seekpolicy_baselinetable_experiments")
DEFAULT_K_VALUES = "1,5,10"
DEFAULT_WINDOWS = "1,2,5,10"
DEFAULT_ENCODER = "text-embedding-3-small"
DEFAULT_QUERY_COLUMN = "RAG_v1_summary"
DEFAULT_DOC_ID_COLUMN = "Document ID"
DEFAULT_FOLD_COLUMN = "fold"
DEFAULT_CHROMA_DIR = Path("data/vectorstore/policy_chunks_chroma")
DEFAULT_CHROMA_COLLECTION = "policy_chunks_openai"
DEFAULT_RAG_RETRIEVER_TOP_K = 1000
DEFAULT_OWID_DIR = Path("data/owid")
DEBUG_ENV_VAR = "SEEKPOLICY_DEBUG"
DEBUG_MAX_QUERIES_ENV_VAR = "SEEKPOLICY_DEBUG_MAX_QUERIES"

_OPENAI_SEMAPHORE: threading.BoundedSemaphore | None = None


def configure_openai_concurrency(limit: int) -> None:
    global _OPENAI_SEMAPHORE
    safe_limit = max(int(limit), 1)
    _OPENAI_SEMAPHORE = threading.BoundedSemaphore(safe_limit)


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "rate limit" in text
        or "too many requests" in text
        or "429" in text
        or "quota" in text
    )


def call_openai_with_guard(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 5,
    base_delay_seconds: float = 1.0,
    **kwargs: Any,
) -> Any:
    delay = max(base_delay_seconds, 0.0)
    attempts = max(max_retries, 1)

    for attempt in range(attempts):
        semaphore = _OPENAI_SEMAPHORE
        if semaphore is not None:
            acquired = semaphore.acquire(timeout=300)
            if not acquired:
                raise RuntimeError(
                    "OpenAI semaphore acquire timed out after 300s — possible starvation"
                )
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt >= attempts - 1 or not _is_rate_limit_error(exc):
                raise
        finally:
            if semaphore is not None:
                semaphore.release()

        if delay > 0:
            time.sleep(delay)
        delay = min(delay * 2 if delay > 0 else 1.0, 16.0)


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


def parse_windows(raw: str) -> list[int]:
    windows = parse_csv_ints(raw)
    if any(window <= 0 for window in windows):
        raise ValueError("All window values must be positive integers")
    return windows


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def format_numeric_cell(cell: str) -> str:
    token = clean_text(cell)
    if not token:
        return ""

    if "=" in token:
        key, value = token.split("=", 1)
        key = clean_text(key)
        value = clean_text(value)
        try:
            return f"{key}={float(value):.2f}"
        except Exception:
            return token

    try:
        return f"{float(token):.2f}"
    except Exception:
        return token


def format_table_row(row: str) -> str:
    row_text = clean_text(row)
    if not row_text:
        return ""
    cells = [format_numeric_cell(cell) for cell in row_text.split(",")]
    cells = [cell for cell in cells if cell]
    return ",".join(cells)


def format_table_block(table_title: str, rows: list[str]) -> str:
    cleaned_rows = [format_table_row(row) for row in rows]
    cleaned_rows = [row for row in cleaned_rows if row]
    if not cleaned_rows:
        return ""
    prefixed_rows = "\n".join(f"[ROW] {row}" for row in cleaned_rows)
    return "[START_OF_TABLE]\n" f"{table_title}\n" f"{prefixed_rows}\n" "[END_OF_TABLE]"


def ensure_openai_api_key_loaded() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("OPENAI_API_KEY is not set. Add it to `.env`.")


def is_debug_enabled() -> bool:
    value = clean_text(os.getenv(DEBUG_ENV_VAR, "")).lower()
    return value in {"1", "true", "yes", "y", "on"}


def get_debug_max_queries() -> int:
    raw = clean_text(os.getenv(DEBUG_MAX_QUERIES_ENV_VAR, "1"))
    try:
        return max(int(raw), 1)
    except Exception:
        return 1


def debug_log_prompt(step: str, prompt: str) -> None:
    print(f"\n[DEBUG][PROMPT][{step}]\n{prompt}\n")


def debug_log_response(step: str, response: str) -> None:
    print(f"\n[DEBUG][RESPONSE][{step}]\n{response}\n")


def load_policy_df(
    path: Path,
    fold_col: str,
    doc_id_col: str,
    query_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {fold_col, doc_id_col}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    out = df.copy()
    out[doc_id_col] = out[doc_id_col].astype(str)
    out[fold_col] = out[fold_col].astype(int)

    if query_col not in out.columns:
        out[query_col] = ""
    out[query_col] = out[query_col].apply(clean_text)

    out = out[out[doc_id_col] != ""]
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


class OpenAISemanticRetriever:
    def __init__(self, corpus_texts: list[str], doc_ids: list[str], model_name: str):
        ensure_openai_api_key_loaded()
        self.embeddings = OpenAIEmbeddings(model=model_name)
        self.corpus_texts = corpus_texts
        self.doc_ids = doc_ids

        if corpus_texts:
            self.corpus_embeddings = np.asarray(
                call_openai_with_guard(self.embeddings.embed_documents, corpus_texts),
                dtype=np.float64,
            )
        else:
            self.corpus_embeddings = np.zeros((0, 0), dtype=np.float64)

    def retrieve(
        self,
        query: str,
        k: int,
        country: str = "",
        sector: str = "",
    ) -> list[str]:
        if not self.corpus_texts or self.corpus_embeddings.size == 0:
            return []

        query_embedding = np.asarray(
            [call_openai_with_guard(self.embeddings.embed_query, query)],
            dtype=np.float64,
        )
        scores = cosine_similarity(query_embedding, self.corpus_embeddings)[0].astype(
            np.float64
        )

        ranked = rank_from_scores(scores, self.doc_ids)
        return ranked[:k]


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


def metadata_get_int(metadata: dict, keys: list[str]) -> int | None:
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


def discover_owid_table_catalog(owid_dir: Path) -> tuple[dict[str, Path], list[str]]:
    table_paths: dict[str, Path] = {}
    table_titles: list[str] = []
    if not owid_dir.exists():
        return table_paths, table_titles

    for file_path in sorted(owid_dir.rglob("*.meta.json")):
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                obj = json.load(handle)
        except Exception:
            continue

        title = ""
        if isinstance(obj, dict):
            raw_title = obj.get("title", None)
            if isinstance(raw_title, str) and raw_title.strip():
                title = raw_title.strip().lower()
            else:
                dataset = obj.get("dataset", {})
                if isinstance(dataset, dict):
                    dataset_title = dataset.get("title", None)
                    if isinstance(dataset_title, str) and dataset_title.strip():
                        title = dataset_title.strip().lower()

        if not title:
            continue

        table_paths[title] = file_path
        table_titles.append(title)

    return table_paths, table_titles


def parse_selected_table_titles(
    response_text: str, known_titles: set[str]
) -> list[str]:
    text = clean_text(response_text)
    if not text:
        return []

    candidates: list[str] = []
    try:
        parsed = literal_eval(response_text)
        if isinstance(parsed, (list, tuple, set)):
            candidates = [clean_text(str(item)).lower() for item in parsed]
    except Exception:
        pass

    if not candidates:
        normalized = response_text.replace("\n", ",").replace("•", ",")
        normalized = normalized.replace(" - ", ",").replace(";", ",")
        split_tokens = [token.strip().lower() for token in normalized.split(",")]
        candidates = [token for token in split_tokens if token]

    matched_by_substring: list[str] = []
    lowered_response = response_text.lower()
    for title in known_titles:
        if title in lowered_response:
            matched_by_substring.append(title)
    candidates.extend(matched_by_substring)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        item_clean = clean_text(item).lower()
        if item_clean in known_titles and item_clean not in seen:
            deduped.append(item_clean)
            seen.add(item_clean)
            continue
        for title in known_titles:
            if item_clean and item_clean in title and title not in seen:
                deduped.append(title)
                seen.add(title)
    return deduped


def extract_policy_year(row: pd.Series) -> int | None:
    raw = clean_text(row.get("Last event in timeline", ""))
    if not raw:
        return None
    try:
        return int(raw.split("-")[0])
    except Exception:
        return None


def retrieve_data_as_string(frame: pd.DataFrame, location: str, input_year: int) -> str:
    loc = clean_text(location)
    if not loc:
        return ""

    if "country" in frame.columns:
        if loc == "European Union":
            filtered = frame[
                frame["country"].astype(str).str.contains("European Union", na=False)
            ]
        else:
            filtered = frame[frame["country"].astype(str) == loc]
    elif "location" in frame.columns:
        filtered = frame[frame["location"].astype(str) == loc]
    else:
        return ""

    if "year" not in filtered.columns:
        return ""

    filtered = filtered[
        (filtered["year"] >= input_year - 10) & (filtered["year"] <= input_year)
    ]
    if filtered.empty:
        return ""

    try:
        return filtered.to_string(
            index=False, float_format=lambda value: f"{value:.2f}"
        )
    except Exception:
        return filtered.to_string(index=False)


def retrieve_top_chunks_for_query(
    query: str,
    country: str,
    sector: str,
    vector_store: object,
    allowed_doc_ids: set[str],
    top_docs: int = 3,
) -> list[str]:
    debug_mode = is_debug_enabled()

    def get_doc_country(metadata: dict) -> str:
        return metadata_get_str(
            metadata,
            ["geography", "country", "Geography", "Country", "location", "Location"],
        )

    candidate_filters: list[dict[str, str]] = []
    if country and sector:
        candidate_filters.append({"geography": country, "sector": sector})
    if country:
        candidate_filters.append({"geography": country})
    candidate_filters.append({})

    docs: list = []
    selected_filter: dict[str, str] | None = None
    for filter_query in candidate_filters:
        try:
            docs = call_openai_with_guard(
                vector_store.similarity_search_with_score,
                query,
                k=DEFAULT_RAG_RETRIEVER_TOP_K,
                filter=filter_query or None,
            )
        except Exception:
            docs = []
        if debug_mode:
            active_filter = filter_query if filter_query else {}
            print(
                f"[DEBUG][RETRIEVE] filter_try={active_filter} | candidates={len(docs)}"
            )
        if docs:
            selected_filter = filter_query
            break

    if debug_mode:
        used_filter = selected_filter if selected_filter is not None else {}
        print(
            f"[DEBUG][RETRIEVE] selected_filter={used_filter} | requested_country={country} | requested_sector={sector}"
        )

    chunks: list[str] = []
    seen_doc_ids: set[str] = set()
    for item in docs:
        doc = item[0]
        metadata = getattr(doc, "metadata", {}) or {}
        doc_id = clean_text(metadata.get("document_id", ""))
        if doc_id and doc_id not in allowed_doc_ids:
            continue
        if doc_id and doc_id in seen_doc_ids:
            continue
        chunk = clean_text(getattr(doc, "page_content", ""))
        if not chunk:
            continue

        if debug_mode:
            doc_country = get_doc_country(metadata)
            country_match = bool(country) and clean_text(doc_country) == clean_text(
                country
            )
            print(
                f"[DEBUG][RETRIEVE] selected_doc_id={doc_id or '<missing_doc_id>'} | country={doc_country or '<missing_country>'} | country_match={country_match}"
            )

        chunks.append(chunk)
        if doc_id:
            seen_doc_ids.add(doc_id)
        if len(seen_doc_ids) >= top_docs:
            break

    return chunks


def load_fold_chunk_corpus(
    chroma_dir: Path,
    collection_name: str,
    fold_id: int,
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    np.ndarray | None,
]:
    try:
        from langchain_chroma import Chroma
    except Exception:
        from langchain_community.vectorstores import Chroma

    if not chroma_dir.exists():
        raise FileNotFoundError(f"Persisted Chroma directory not found: {chroma_dir}.")

    store = Chroma(
        collection_name=collection_name,
        embedding_function=None,
        persist_directory=str(chroma_dir),
    )

    page_size = 1000
    offset = 0
    chunk_texts: list[str] = []
    chunk_doc_ids: list[str] = []
    chunk_geographies: list[str] = []
    chunk_sectors: list[str] = []
    chunk_ids: list[str] = []
    chunk_sources: list[str] = []
    chunk_embeddings_raw: list[object] = []

    while True:
        payload = store.get(
            include=["documents", "metadatas", "embeddings"],
            limit=page_size,
            offset=offset,
        )
        documents = payload.get("documents", []) or []
        metadatas = payload.get("metadatas", []) or []
        embeddings = payload.get("embeddings", None)

        if not documents:
            break

        if embeddings is None:
            embeddings = []
        elif isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        if len(embeddings) != len(documents):
            embeddings = [None] * len(documents)

        for text, metadata, embedding in zip(documents, metadatas, embeddings):
            metadata = metadata or {}
            item_fold = metadata_get_int(metadata, ["fold", "Fold"])
            if item_fold != int(fold_id):
                continue

            doc_id = metadata_get_str(
                metadata,
                ["document_id", "Document ID", "doc_id", "documentId"],
            )
            cleaned_text = clean_text(text)
            if not doc_id or not cleaned_text:
                continue

            chunk_texts.append(cleaned_text)
            chunk_doc_ids.append(doc_id)
            chunk_geographies.append(
                metadata_get_str(
                    metadata,
                    [
                        "geography",
                        "country",
                        "Geography",
                        "Country",
                        "location",
                        "Location",
                    ],
                )
            )
            chunk_sectors.append(
                metadata_get_str(metadata, ["sector", "Sector", "topic", "Topic"])
            )
            chunk_ids.append(metadata_get_str(metadata, ["chunk_id", "chunkId"]))
            chunk_sources.append(
                metadata_get_str(metadata, ["source", "file_path", "path"])
            )
            chunk_embeddings_raw.append(embedding)

        if len(documents) < page_size:
            break
        offset += page_size

    if not chunk_texts:
        raise RuntimeError(
            f"No chunk entries found for fold={fold_id} in collection `{collection_name}` at {chroma_dir}."
        )

    precomputed_embeddings: np.ndarray | None = None
    if chunk_embeddings_raw and all(
        isinstance(embedding, (list, tuple)) and len(embedding) > 0
        for embedding in chunk_embeddings_raw
    ):
        try:
            precomputed_embeddings = np.asarray(chunk_embeddings_raw, dtype=np.float64)
        except Exception:
            precomputed_embeddings = None

    return (
        chunk_texts,
        chunk_doc_ids,
        chunk_geographies,
        chunk_sectors,
        chunk_ids,
        chunk_sources,
        precomputed_embeddings,
    )


class OpenAIChunkSemanticRetriever:
    def __init__(
        self,
        chunk_texts: list[str],
        chunk_doc_ids: list[str],
        chunk_geographies: list[str],
        chunk_sectors: list[str],
        chunk_ids: list[str],
        chunk_sources: list[str],
        model_name: str,
        use_metadata_filter: bool = False,
        embedding_batch_size: int = 128,
        precomputed_chunk_embeddings: np.ndarray | None = None,
    ):
        ensure_openai_api_key_loaded()
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be > 0")

        self.embeddings = OpenAIEmbeddings(model=model_name)
        self.chunk_texts = chunk_texts
        self.chunk_doc_ids = chunk_doc_ids
        self.chunk_geographies = [clean_text(value) for value in chunk_geographies]
        self.chunk_sectors = [clean_text(value) for value in chunk_sectors]
        self.chunk_ids = [clean_text(value) for value in chunk_ids]
        self.chunk_sources = [clean_text(value) for value in chunk_sources]
        self.use_metadata_filter = bool(use_metadata_filter)

        if len(self.chunk_geographies) != len(self.chunk_texts):
            raise ValueError("chunk_geographies length must match chunk_texts length")
        if len(self.chunk_sectors) != len(self.chunk_texts):
            raise ValueError("chunk_sectors length must match chunk_texts length")
        if len(self.chunk_ids) != len(self.chunk_texts):
            raise ValueError("chunk_ids length must match chunk_texts length")
        if len(self.chunk_sources) != len(self.chunk_texts):
            raise ValueError("chunk_sources length must match chunk_texts length")

        if precomputed_chunk_embeddings is not None:
            if precomputed_chunk_embeddings.shape[0] != len(chunk_texts):
                raise ValueError(
                    "precomputed_chunk_embeddings row count must match chunk_texts length"
                )
            self.chunk_embeddings = np.asarray(
                precomputed_chunk_embeddings, dtype=np.float64
            )
        else:
            vectors: list[list[float]] = []
            if chunk_texts:
                for start in range(0, len(chunk_texts), embedding_batch_size):
                    batch = chunk_texts[start : start + embedding_batch_size]
                    vectors.extend(
                        call_openai_with_guard(self.embeddings.embed_documents, batch)
                    )
            self.chunk_embeddings = np.asarray(vectors, dtype=np.float64)

    def score_chunk_candidates(
        self,
        query: str,
        country: str = "",
        sector: str = "",
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        if not self.chunk_texts:
            return np.zeros(0, dtype=np.float64), [], np.asarray([], dtype=np.int64)

        candidate_indices = np.arange(len(self.chunk_texts), dtype=np.int64)
        debug_mode = is_debug_enabled()

        if self.use_metadata_filter:
            requested_country = clean_text(country)
            requested_sector = clean_text(sector)

            def select_indices(filter_country: str, filter_sector: str) -> np.ndarray:
                selected: list[int] = []
                for idx, (geo, sec) in enumerate(
                    zip(self.chunk_geographies, self.chunk_sectors)
                ):
                    if filter_country and geo != filter_country:
                        continue
                    if filter_sector and sec != filter_sector:
                        continue
                    selected.append(idx)
                return np.asarray(selected, dtype=np.int64)

            filter_plan: list[tuple[str, str]] = []
            if requested_country and requested_sector:
                filter_plan.append((requested_country, requested_sector))
            if requested_country:
                filter_plan.append((requested_country, ""))
            filter_plan.append(("", ""))

            selected_filter = ("", "")
            candidate_indices = np.asarray([], dtype=np.int64)
            for f_country, f_sector in filter_plan:
                trial = select_indices(f_country, f_sector)
                if debug_mode:
                    print(
                        "[DEBUG][EVAL_RETRIEVE] "
                        f"filter_try={{'geography': {f_country!r}, 'sector': {f_sector!r}}} | candidates={len(trial)}"
                    )
                if len(trial) > 0:
                    candidate_indices = trial
                    selected_filter = (f_country, f_sector)
                    break

            if debug_mode:
                print(
                    "[DEBUG][EVAL_RETRIEVE] "
                    f"selected_filter={{'geography': {selected_filter[0]!r}, 'sector': {selected_filter[1]!r}}}"
                )

            if len(candidate_indices) == 0:
                return np.zeros(0, dtype=np.float64), [], np.asarray([], dtype=np.int64)
        elif debug_mode:
            print("[DEBUG][EVAL_RETRIEVE] metadata_filter=OFF")

        query_embedding = np.asarray(
            [call_openai_with_guard(self.embeddings.embed_query, query)],
            dtype=np.float64,
        )
        filtered_embeddings = self.chunk_embeddings[candidate_indices]
        filtered_doc_ids = [self.chunk_doc_ids[idx] for idx in candidate_indices]

        scores = cosine_similarity(query_embedding, filtered_embeddings)[0].astype(
            np.float64
        )
        return scores, filtered_doc_ids, candidate_indices

    def retrieve(
        self,
        query: str,
        k: int,
        country: str = "",
        sector: str = "",
    ) -> list[str]:
        scores, filtered_doc_ids, _ = self.score_chunk_candidates(
            query=query,
            country=country,
            sector=sector,
        )
        if len(scores) == 0:
            return []

        ranked = rank_from_scores(scores, filtered_doc_ids)
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


def mrr_at_k(ranked: list[str], target: str, k: int) -> float:
    if target not in ranked[:k]:
        return 0.0
    rank = ranked[:k].index(target) + 1
    return 1.0 / rank


def evaluate_fold(
    fold_id: int,
    window: int,
    method_name: str,
    fold_df: pd.DataFrame,
    query_col: str,
    doc_id_col: str,
    retriever: OpenAISemanticRetriever | OpenAIChunkSemanticRetriever,
    k_values: list[int],
    save_retrieval_traces: bool,
    retrieval_trace_top_k: int,
    retrieval_trace_output_path: Path | None,
) -> dict[str, float]:
    records = {f"hit@{k}": [] for k in k_values}
    records.update({f"precision@{k}": [] for k in k_values})
    records.update({f"ndcg@{k}": [] for k in k_values})
    records.update({f"mrr@{k}": [] for k in k_values})

    skipped = 0
    max_k = max(k_values)
    trace_top_k = retrieval_trace_top_k if retrieval_trace_top_k > 0 else max_k
    retrieval_trace_rows: list[dict[str, object]] = []
    debug_mode = is_debug_enabled()
    debug_max_queries = get_debug_max_queries()
    debug_seen = 0

    for query_index, row in fold_df.iterrows():
        query = clean_text(row.get(query_col, ""))
        target = str(row.get(doc_id_col, ""))
        country = clean_text(row.get("Geography", ""))
        sector = clean_text(row.get("Sector", ""))
        if debug_mode and debug_seen < debug_max_queries:
            print(f"[DEBUG][EVAL] test_doc_id={target} | query_present={bool(query)}")
        if not query or not target:
            skipped += 1
            if debug_mode and debug_seen < debug_max_queries:
                print(
                    "[DEBUG][EVAL] retrieved_doc_ids_top= [] (skipped: empty query/target)"
                )
                debug_seen += 1
            continue

        ranked: list[str] = []
        if save_retrieval_traces and isinstance(
            retriever, OpenAIChunkSemanticRetriever
        ):
            chunk_scores, filtered_doc_ids, candidate_indices = (
                retriever.score_chunk_candidates(
                    query=query,
                    country=country,
                    sector=sector,
                )
            )
            if len(chunk_scores) > 0:
                ranked = rank_from_scores(chunk_scores, filtered_doc_ids)

                chunk_rank_order = np.argsort(-chunk_scores)
                max_rows = min(trace_top_k, len(chunk_rank_order))
                for rank in range(max_rows):
                    idx_in_filtered = int(chunk_rank_order[rank])
                    chunk_idx = int(candidate_indices[idx_in_filtered])
                    retrieved_doc_id = str(filtered_doc_ids[idx_in_filtered])
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
                            "retrieved_score": float(chunk_scores[idx_in_filtered]),
                            "is_relevant_doc": float(retrieved_doc_id == target),
                            "retrieved_chunk_id": retriever.chunk_ids[chunk_idx],
                            "retrieved_chunk_source": retriever.chunk_sources[
                                chunk_idx
                            ],
                            "retrieved_chunk_text": retriever.chunk_texts[chunk_idx],
                        }
                    )
        else:
            ranked = retriever.retrieve(
                query=query, k=max_k, country=country, sector=sector
            )

        if not ranked:
            skipped += 1
            if debug_mode and debug_seen < debug_max_queries:
                print("[DEBUG][EVAL] retrieved_doc_ids_top= [] (no retrieval results)")
                debug_seen += 1
            continue

        if debug_mode and debug_seen < debug_max_queries:
            print(
                f"[DEBUG][EVAL] test_doc_id={target} | retrieved_doc_ids_top_{max_k}={ranked[:max_k]}"
            )
            debug_seen += 1

        for k in k_values:
            records[f"hit@{k}"].append(hit_at_k(ranked, target, k))
            records[f"precision@{k}"].append(precision_at_k(ranked, target, k))
            records[f"ndcg@{k}"].append(ndcg_at_k(ranked, target, k))
            records[f"mrr@{k}"].append(mrr_at_k(ranked, target, k))

    metrics: dict[str, float] = {}
    for name, values in records.items():
        metrics[name] = float(np.mean(values)) if values else 0.0

    metrics["queries_evaluated"] = float(max(len(fold_df) - skipped, 0))
    metrics["queries_total"] = float(len(fold_df))

    if (
        save_retrieval_traces
        and retrieval_trace_output_path is not None
        and retrieval_trace_rows
    ):
        retrieval_trace_output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(retrieval_trace_rows).to_csv(
            retrieval_trace_output_path, index=False
        )

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


def build_window_test_dataframe(
    policy_df: pd.DataFrame,
    fold_id: int,
    window_records: list[dict],
    fold_col: str,
    doc_id_col: str,
    query_col: str,
) -> pd.DataFrame:
    fold_policy = policy_df[policy_df[fold_col] == fold_id].copy()
    if fold_policy.empty:
        return pd.DataFrame(columns=[doc_id_col, query_col, "Geography", "Sector"])

    doc_to_query: dict[str, str] = {}
    doc_to_country: dict[str, str] = {}
    doc_to_sector: dict[str, str] = {}
    for _, row in fold_policy.iterrows():
        doc_id = str(row.get(doc_id_col, "")).strip()
        if not doc_id:
            continue
        query_text = clean_text(row.get(query_col, ""))
        if query_text and doc_id not in doc_to_query:
            doc_to_query[doc_id] = query_text
        if doc_id not in doc_to_country:
            doc_to_country[doc_id] = clean_text(row.get("Geography", ""))
        if doc_id not in doc_to_sector:
            doc_to_sector[doc_id] = clean_text(row.get("Sector", ""))

    rows: list[dict[str, str]] = []
    for record in window_records:
        target_doc_id = str(record.get("doc_id", "")).strip()
        if not target_doc_id:
            continue
        rows.append(
            {
                doc_id_col: target_doc_id,
                query_col: doc_to_query.get(target_doc_id, ""),
                "Geography": doc_to_country.get(target_doc_id, ""),
                "Sector": doc_to_sector.get(target_doc_id, ""),
            }
        )

    if not rows:
        return pd.DataFrame(columns=[doc_id_col, query_col, "Geography", "Sector"])

    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def collect_window_test_doc_ids(
    window_records: list[dict],
    fold_id: int,
    window: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in window_records:
        target_doc_id = clean_text(record.get("doc_id", ""))
        if not target_doc_id:
            continue
        rows.append(
            {
                "fold": int(fold_id),
                "window": int(window),
                "doc_id": target_doc_id,
            }
        )
    return rows


def load_existing_chroma_store(
    chroma_dir: Path,
    collection_name: str,
) -> object:
    try:
        from langchain_chroma import Chroma
    except Exception:
        from langchain_community.vectorstores import Chroma

    try:
        from langchain_openai import OpenAIEmbeddings
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Missing dependencies for Chroma retrieval. Install with `pip install langchain-openai langchain-community langchain-chroma chromadb`."
        ) from exc

    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"Persisted Chroma directory not found: {chroma_dir}. "
            "Build it first before running generation."
        )

    embeddings = OpenAIEmbeddings()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )


def generate_rag_summaries(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    doc_id_col: str,
    model_name: str,
    rag_k: int,
    chroma_dir: Path,
    chroma_collection: str,
    allowed_doc_ids: set[str],
    history_window: int,
    debug_mode: bool = False,
    debug_target_doc_ids: set[str] | None = None,
    progress_prefix: str = "",
    progress_every: int = 20,
    verbose_progress: bool = True,
    intra_doc_parallelism: int = 1,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    ensure_openai_api_key_loaded()

    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Missing dependencies for RAG generation. Install with `pip install langchain-openai langchain-community langchain-chroma chromadb`."
        ) from exc

    vector_store = load_existing_chroma_store(
        chroma_dir=chroma_dir,
        collection_name=chroma_collection,
    )

    llm = ChatOpenAI(model=model_name, temperature=0)
    table_names, table_titles = discover_owid_table_catalog(DEFAULT_OWID_DIR)
    known_titles = set(table_titles)

    # ------------------------------------------------------------------
    # Per-doc checkpoint
    # ------------------------------------------------------------------
    _ckpt_lock = threading.Lock()

    def _load_doc_checkpoint() -> dict[str, str]:
        if checkpoint_path is None or not checkpoint_path.exists():
            return {}
        try:
            ckpt_df = pd.read_csv(checkpoint_path, dtype=str).fillna("")
            if doc_id_col not in ckpt_df.columns or target_col not in ckpt_df.columns:
                return {}
            return dict(zip(ckpt_df[doc_id_col], ckpt_df[target_col]))
        except Exception as exc:
            print(
                f"[GEN][CHECKPOINT_WARN] Could not read doc checkpoint {checkpoint_path}: {exc}",
                flush=True,
            )
            return {}

    def _append_doc_checkpoint(doc_id: str, text: str) -> None:
        if checkpoint_path is None:
            return
        try:
            with _ckpt_lock:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                row_df = pd.DataFrame([{doc_id_col: doc_id, target_col: text}])
                write_header = not checkpoint_path.exists()
                row_df.to_csv(
                    checkpoint_path, mode="a", header=write_header, index=False
                )
        except Exception as exc:
            print(
                f"[GEN][CHECKPOINT_WARN] Could not append doc checkpoint for doc_id={doc_id}: {exc}",
                flush=True,
            )

    completed_doc_map = _load_doc_checkpoint()

    if history_window <= 0:
        raise ValueError("history_window must be > 0")

    intra_workers = max(int(intra_doc_parallelism), 1)

    out = df.copy()
    if target_col not in out.columns:
        out[target_col] = ""

    # Apply checkpoint results now that out is initialised
    if completed_doc_map:
        for row_idx in out.index:
            did = clean_text(out.at[row_idx, doc_id_col])
            if did in completed_doc_map:
                saved = completed_doc_map[did]
                if saved:
                    out.at[row_idx, target_col] = saved
        if verbose_progress:
            print(
                f"[GEN][CHECKPOINT_LOADED] {progress_prefix} resumed={len(completed_doc_map)} docs from {checkpoint_path}",
                flush=True,
            )

    total_docs = len(out.index)
    processed_docs = 0
    generated_docs = 0
    progress_interval = max(int(progress_every), 1)
    if verbose_progress:
        print(
            f"[GEN][START] {progress_prefix} docs={total_docs} history_window={history_window} intra_doc_parallelism={intra_workers}",
            flush=True,
        )

    def retrieve_data_as_string_for_window(
        frame: pd.DataFrame, location: str, test_year: int
    ) -> str:
        loc = clean_text(location)
        if not loc:
            return ""

        if "country" in frame.columns:
            if loc == "European Union":
                filtered = frame[
                    frame["country"]
                    .astype(str)
                    .str.contains("European Union", na=False)
                ]
            else:
                filtered = frame[frame["country"].astype(str) == loc]
        elif "location" in frame.columns:
            filtered = frame[frame["location"].astype(str) == loc]
        else:
            return ""

        if "year" not in filtered.columns:
            return ""

        end_year = int(test_year) - 1
        start_year = end_year - int(history_window) + 1
        filtered = filtered[
            (filtered["year"] >= start_year) & (filtered["year"] <= end_year)
        ]
        if filtered.empty:
            return ""

        try:
            return filtered.to_string(
                index=False, float_format=lambda value: f"{value:.2f}"
            )
        except Exception:
            return filtered.to_string(index=False)

    # ------------------------------------------------------------------
    # analyse_one_title: Doc 2 style — flat string concatenation, same prompt
    # ------------------------------------------------------------------
    def analyse_one_title(
        title: str, sector: str, country: str, region: str, policy_year: int | None
    ) -> str | None:
        file_path = table_names.get(title)
        if file_path is None:
            return None

        try:
            with file_path.open("r", encoding="utf-8") as handle:
                meta_obj = json.load(handle)
        except Exception:
            return None

        # Build dataset text as a flat string (Doc 2 style)
        text = "DATASET:\n"
        dataset = meta_obj if isinstance(meta_obj, dict) else {}
        desc = None
        if "dataset" in dataset and isinstance(dataset["dataset"], dict):
            desc = dataset["dataset"].get("description", None)
        elif "description" in dataset:
            desc = dataset.get("description", None)

        text += f"Title: {title}\n"
        if desc:
            text += f"Description: {desc}\n"

        csv_path = Path(str(file_path).replace(".meta.json", ".csv"))
        if not csv_path.exists():
            return None

        try:
            dataset_frame = pd.read_csv(csv_path)
        except Exception:
            return None

        if country and policy_year is not None:
            text += f"Focused Country Data: {country}\n"
            country_data = retrieve_data_as_string_for_window(
                dataset_frame, country, policy_year
            )
            text += country_data

        if region and policy_year is not None:
            text += f"\n{region} Data:\n"
            region_data = retrieve_data_as_string_for_window(
                dataset_frame, region, policy_year
            )
            text += region_data

        if policy_year is not None:
            text += "\nWorld Data:\n"
            world_data = retrieve_data_as_string_for_window(
                dataset_frame, "World", policy_year
            )
            text += world_data

        if not text.strip():
            return None

        try:
            analysis_prompt = f"""
        You are policy analyzer expert.
        You are to analyse the following data, but keep in mind that you are focused at {sector} aspects.
        Here is the data you will be analysed.

        {text}

        ----

        Output only one paragraph with no explanation.
        Begin:
        """
            if debug_mode:
                debug_log_prompt(f"dataset_analysis::{title}", analysis_prompt)
            analysis_response = call_openai_with_guard(llm.invoke, analysis_prompt)
            step_query = clean_text(analysis_response.content)
            if debug_mode:
                debug_log_response(f"dataset_analysis::{title}", step_query)
            return step_query if step_query else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # blend_one_query: Doc 2 style — simple blend instruction, no policy context injection
    # ------------------------------------------------------------------
    def blend_one_query(
        step_query: str,
        country: str,
        sector: str,
        keyword: str,
        topic: str,
    ) -> str | None:
        related_chunks = retrieve_top_chunks_for_query(
            query=step_query,
            country=country,
            sector=sector,
            vector_store=vector_store,
            allowed_doc_ids=allowed_doc_ids,
            top_docs=3,
        )
        if not related_chunks:
            return None

        instruction = (
            "Generate a single coherent paragraph based on the following time series analysis and related content. "
            "Ensure that the writing style of the generated paragraph is a blend of the three other chunks, "
            "while incorporating the insights from the time series analysis. "
            "Do not mention the name of any policy, plan, act, or refer to 'time series analysis' explicitly."
        )

        input_text = (
            instruction
            + "\n\nTime Series Analysis Chunk:\n"
            + step_query
            + "\n\nOther Related Chunks:\n"
            + "\n".join(related_chunks)
        )

        try:
            if debug_mode:
                debug_log_prompt("chunk_blend", input_text)
            new_chunk = call_openai_with_guard(llm.invoke, input_text)
            new_text = clean_text(new_chunk.content)
            if debug_mode:
                debug_log_response("chunk_blend", new_text)
            return new_text if new_text else None
        except Exception:
            return None

    for idx in out.index:
        processed_docs += 1
        current_doc_id = clean_text(out.at[idx, doc_id_col])
        if (
            debug_target_doc_ids is not None
            and current_doc_id not in debug_target_doc_ids
        ):
            if verbose_progress and (
                processed_docs % progress_interval == 0 or processed_docs == total_docs
            ):
                print(
                    f"[GEN][PROGRESS] {progress_prefix} processed={processed_docs}/{total_docs} generated={generated_docs}",
                    flush=True,
                )
            continue

        existing = clean_text(out.at[idx, target_col])
        if existing:
            if verbose_progress and (
                processed_docs % progress_interval == 0 or processed_docs == total_docs
            ):
                print(
                    f"[GEN][PROGRESS] {progress_prefix} processed={processed_docs}/{total_docs} generated={generated_docs}",
                    flush=True,
                )
            continue

        sector = clean_text(out.at[idx, "Sector"]) if "Sector" in out.columns else ""
        instrument = (
            clean_text(out.at[idx, "Instrument"]) if "Instrument" in out.columns else ""
        )
        keyword = clean_text(out.at[idx, "Keyword"]) if "Keyword" in out.columns else ""
        topic = (
            clean_text(out.at[idx, "Topic/Response"])
            if "Topic/Response" in out.columns
            else ""
        )
        hazard = clean_text(out.at[idx, "Hazard"]) if "Hazard" in out.columns else ""
        country = (
            clean_text(out.at[idx, "Geography"]) if "Geography" in out.columns else ""
        )
        region = clean_text(out.at[idx, "region"]) if "region" in out.columns else ""
        policy_year = extract_policy_year(out.loc[idx])

        policy_metadata = (
            f"Targeted Sector: {sector}\n"
            f"Policy Instrument: {instrument}\n"
            f"Keywords: {keyword}\n"
            f"Topics: {topic}\n"
            f"Hazards: {hazard}"
        )
        if not policy_metadata.strip():
            continue

        # Table selection prompt (Doc 2 style)
        selected_tables_prompt = f"""
        Based on this table's names: {table_titles}, I want you to select all the table's names from the list that might be directly related to the creation of the policy with the following metadata.

        POLICY METADATA:
        {policy_metadata}

        Return in list with no explaination.
        Begin:
        """
        if debug_mode:
            debug_log_prompt("table_selector", selected_tables_prompt)
        selected_tables_response = call_openai_with_guard(
            llm.invoke, selected_tables_prompt
        )
        if debug_mode:
            debug_log_response(
                "table_selector", clean_text(selected_tables_response.content)
            )
        selected_titles = parse_selected_table_titles(
            str(selected_tables_response.content), known_titles
        )

        # --- PARALLEL: analyse all selected tables concurrently ---
        step_1_queries: list[str] = []
        if intra_workers > 1 and len(selected_titles) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=intra_workers
            ) as tex:
                title_futures = {
                    tex.submit(
                        analyse_one_title, title, sector, country, region, policy_year
                    ): title
                    for title in selected_titles
                }
                ordered_results: dict[str, str | None] = {}
                for future in concurrent.futures.as_completed(title_futures):
                    title = title_futures[future]
                    try:
                        ordered_results[title] = future.result()
                    except Exception:
                        ordered_results[title] = None
            for title in selected_titles:
                result = ordered_results.get(title)
                if result:
                    step_1_queries.append(result)
        else:
            for title in selected_titles:
                result = analyse_one_title(title, sector, country, region, policy_year)
                if result:
                    step_1_queries.append(result)

        if not step_1_queries:
            fallback_source = ""
            if source_col != target_col:
                fallback_source = clean_text(out.at[idx, source_col])
            if fallback_source:
                step_1_queries.append(fallback_source)

        # --- PARALLEL: blend all step queries concurrently ---
        all_new_chunks: list[str] = []
        if intra_workers > 1 and len(step_1_queries) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=intra_workers
            ) as tex:
                blend_futures = {
                    tex.submit(
                        blend_one_query, step_query, country, sector, keyword, topic
                    ): i
                    for i, step_query in enumerate(step_1_queries)
                }
                ordered_chunks: dict[int, str | None] = {}
                for future in concurrent.futures.as_completed(blend_futures):
                    i = blend_futures[future]
                    try:
                        ordered_chunks[i] = future.result()
                    except Exception:
                        ordered_chunks[i] = None
            for i in range(len(step_1_queries)):
                result = ordered_chunks.get(i)
                if result:
                    all_new_chunks.append(result)
        else:
            for step_query in step_1_queries:
                result = blend_one_query(step_query, country, sector, keyword, topic)
                if result:
                    all_new_chunks.append(result)

        # Deduplicate (Doc 2 uses set(); preserve insertion order here)
        unique_new_chunks = list(dict.fromkeys(all_new_chunks))
        if not unique_new_chunks:
            continue

        # Final summary prompt (Doc 2 style — simple, no sector/keyword framing)
        final_prompt = (
            "You are policy analyzer expert.\n"
            "You are to summarize these chunks of information to one paragraph:\n\n"
            f"{'[NEW_CHUNK]'.join(unique_new_chunks)}\n\n"
            "----\n\n"
            "Output only one paragraph with no explanation.\n"
            "Begin:\n"
        )
        try:
            if debug_mode:
                debug_log_prompt("final_summary", final_prompt)
            final_response = call_openai_with_guard(llm.invoke, final_prompt)
            final_text = clean_text(final_response.content)
            if debug_mode:
                debug_log_response("final_summary", final_text)
            out.at[idx, target_col] = final_text
            if final_text:
                generated_docs += 1
                _append_doc_checkpoint(current_doc_id, final_text)
                if verbose_progress:
                    print(
                        f"[GEN][DOC_SAVED] {progress_prefix} doc_id={current_doc_id} → checkpoint",
                        flush=True,
                    )
        except Exception:
            continue

        if verbose_progress and (
            processed_docs % progress_interval == 0 or processed_docs == total_docs
        ):
            print(
                f"[GEN][PROGRESS] {progress_prefix} processed={processed_docs}/{total_docs} generated={generated_docs}",
                flush=True,
            )

    if verbose_progress:
        print(
            f"[GEN][DONE] {progress_prefix} processed={processed_docs}/{total_docs} generated={generated_docs}",
            flush=True,
        )

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SEEK-Policy RAG summary pipeline with persisted Chroma/OpenAI generation, "
            "fold/window-aware retrieval metrics."
        )
    )

    parser.add_argument("--policy-input", type=Path, default=DEFAULT_POLICY_INPUT)
    parser.add_argument("--model-input-dir", type=Path, default=DEFAULT_MODEL_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-values", type=str, default=DEFAULT_K_VALUES)
    parser.add_argument("--windows", type=str, default=DEFAULT_WINDOWS)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument(
        "--fold-parallelism",
        type=int,
        default=1,
        help="Max concurrent fold workers. Used together with --window-parallelism.",
    )
    parser.add_argument(
        "--window-parallelism",
        type=int,
        default=1,
        help="Max concurrent window workers. Used together with --fold-parallelism.",
    )
    parser.add_argument(
        "--intra-doc-parallelism",
        type=int,
        default=1,
        help="Max concurrent workers for table analysis and chunk blend loops within a single document.",
    )
    parser.add_argument(
        "--openai-concurrency",
        type=int,
        default=2,
        help="Global max concurrent OpenAI calls across threads (guarded with retry).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Print generation heartbeat every N processed documents per fold/window worker.",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Reduce non-essential progress logs (keeps fold,window,left/total lines).",
    )

    parser.add_argument("--fold-column", type=str, default=DEFAULT_FOLD_COLUMN)
    parser.add_argument("--doc-id-column", type=str, default=DEFAULT_DOC_ID_COLUMN)
    parser.add_argument("--query-column", type=str, default=DEFAULT_QUERY_COLUMN)

    parser.add_argument(
        "--source-column",
        type=str,
        default="Family Summary",
        help="Source text column used for fallback query text during RAG summary generation.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model used for RAG summary generation.",
    )
    parser.add_argument("--rag-k", type=int, default=5)
    parser.add_argument(
        "--eval-use-metadata-filter",
        action="store_true",
        help="Enable metadata filter (country/sector) for evaluation chunk retrieval. Default is OFF.",
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
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument(
        "--chroma-collection", type=str, default=DEFAULT_CHROMA_COLLECTION
    )
    parser.add_argument(
        "--generated-output",
        type=Path,
        default=None,
        help="Optional CSV path to store generated RAG summaries.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    k_values = parse_csv_ints(args.k_values)
    if any(k <= 0 for k in k_values):
        raise ValueError("All k-values must be positive")
    windows = parse_windows(args.windows)
    if args.window is not None:
        if args.window <= 0:
            raise ValueError("--window must be positive")
        windows = [args.window]
    if args.retrieval_trace_top_k < 0:
        raise ValueError("--retrieval-trace-top-k must be >= 0")
    if args.fold_parallelism <= 0:
        raise ValueError("--fold-parallelism must be > 0")
    if args.window_parallelism <= 0:
        raise ValueError("--window-parallelism must be > 0")
    if args.openai_concurrency <= 0:
        raise ValueError("--openai-concurrency must be > 0")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be > 0")

    policy_df = load_policy_df(
        path=args.policy_input,
        fold_col=args.fold_column,
        doc_id_col=args.doc_id_column,
        query_col=args.query_column,
    )

    ensure_openai_api_key_loaded()
    configure_openai_concurrency(args.openai_concurrency)
    titles = discover_owid_titles(DEFAULT_OWID_DIR)
    embeddings = OpenAIEmbeddings(model=DEFAULT_ENCODER)
    title_vectors = embed_texts(embeddings, titles, batch_size=128)
    title_vectors = normalize_rows(title_vectors)
    # Example: select top-N tables for each policy
    selected_tables_df = select_titles_for_policies(
        policy_df=policy_df,
        titles=titles,
        title_vectors=title_vectors,
        embeddings=embeddings,
        top_n=5,
        batch_size=128,
        fold_col=args.fold_column,
        doc_id_col=args.doc_id_column,
    )
    # Save selected tables
    selected_tables_path = args.output_dir / "selected_tables_embedding.csv"
    selected_tables_df.to_csv(selected_tables_path, index=False)
    print(f"Saved embedding-based selected tables: {selected_tables_path}")

    fold_ids = sorted(policy_df[args.fold_column].unique().tolist())
    if args.fold is not None:
        if args.fold not in fold_ids:
            raise ValueError(
                f"Fold {args.fold} does not exist in {args.policy_input} for column `{args.fold_column}`"
            )
        fold_ids = [args.fold]

    debug_mode = is_debug_enabled()
    debug_max_queries = get_debug_max_queries()
    if debug_mode:
        print(
            f"[DEBUG] Enabled via {DEBUG_ENV_VAR}. Limiting to {debug_max_queries} test query(ies)."
        )

    configure_openai_concurrency(args.openai_concurrency)
    verbose_progress = not args.quiet_progress

    if verbose_progress:
        print(
            "[RUN][CONFIG] "
            f"folds={len(fold_ids)} windows={len(windows)} "
            f"workers={args.fold_parallelism * args.window_parallelism} "
            f"openai_concurrency={args.openai_concurrency}",
            flush=True,
        )

    generated_policy_by_window: dict[int, pd.DataFrame] = {}
    generated_outputs_by_window: dict[str, str] = {}

    for window in windows:
        generated_policy_by_window[int(window)] = policy_df.copy()

    gen_checkpoint_dir = args.output_dir / "gen_checkpoints"
    gen_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run_generation_pair(
        fold_id: int, window: int
    ) -> tuple[int, int, list[int], np.ndarray]:
        if verbose_progress:
            print(
                f"[GEN][PAIR_START] fold={fold_id} window={window}",
                flush=True,
            )
        fold_slice = policy_df[policy_df[args.fold_column] == fold_id].copy()
        row_indices = fold_slice.index.tolist()
        if fold_slice.empty:
            return int(fold_id), int(window), row_indices, np.asarray([])

        allowed_doc_ids = {
            clean_text(value)
            for value in fold_slice[args.doc_id_column].astype(str).tolist()
            if clean_text(value)
        }
        if not allowed_doc_ids:
            return int(fold_id), int(window), row_indices, np.asarray([])

        debug_target_doc_ids: set[str] | None = None
        if debug_mode:
            try:
                dbg_records = load_window_test_records(
                    model_input_dir=args.model_input_dir,
                    fold_id=fold_id,
                    window=window,
                )
                target_ids: list[str] = []
                for record in dbg_records:
                    doc_id = clean_text(record.get("doc_id", ""))
                    if doc_id:
                        target_ids.append(doc_id)
                    if len(target_ids) >= debug_max_queries:
                        break
                if target_ids:
                    debug_target_doc_ids = set(target_ids)
            except Exception:
                debug_target_doc_ids = None

        fold_generated = generate_rag_summaries(
            df=fold_slice,
            source_col=args.source_column,
            target_col=args.query_column,
            doc_id_col=args.doc_id_column,
            intra_doc_parallelism=args.intra_doc_parallelism,
            model_name=args.openai_model,
            rag_k=args.rag_k,
            chroma_dir=args.chroma_dir,
            chroma_collection=args.chroma_collection,
            allowed_doc_ids=allowed_doc_ids,
            history_window=window,
            debug_mode=debug_mode,
            debug_target_doc_ids=debug_target_doc_ids,
            progress_prefix=f"fold={fold_id} window={window}",
            progress_every=args.progress_every,
            verbose_progress=verbose_progress,
            checkpoint_path=args.output_dir
            / "gen_checkpoints"
            / f"gen_fold_{fold_id}_window_{window}.csv",
        )
        values = fold_generated[args.query_column].to_numpy()
        return int(fold_id), int(window), row_indices, values

    generation_pairs = [
        (int(fold_id), int(window)) for fold_id in fold_ids for window in windows
    ]
    total_generation_pairs = len(generation_pairs)
    completed_generation_pairs = 0
    generation_workers = max(
        1,
        min(
            total_generation_pairs,
            int(args.fold_parallelism) * int(args.window_parallelism),
        ),
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=generation_workers
    ) as executor:
        if verbose_progress:
            print(
                f"[GEN][PHASE_START] pairs={total_generation_pairs} workers={generation_workers}",
                flush=True,
            )
        future_to_pair = {
            executor.submit(run_generation_pair, fold_id, window): (fold_id, window)
            for fold_id, window in generation_pairs
        }
        generation_failed_pairs = 0
        pending_generation_futures = set(future_to_pair.keys())
        while pending_generation_futures:
            done_generation_futures, pending_generation_futures = (
                concurrent.futures.wait(
                    pending_generation_futures,
                    timeout=30,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
            )
            if not done_generation_futures:
                if verbose_progress:
                    sample_pending = [
                        future_to_pair[future]
                        for future in list(pending_generation_futures)[:5]
                    ]
                    print(
                        "[GEN][HEARTBEAT] "
                        f"completed={completed_generation_pairs}/{total_generation_pairs} "
                        f"pending={len(pending_generation_futures)} "
                        f"sample_pending={sample_pending}",
                        flush=True,
                    )
                continue

            for future in done_generation_futures:
                pair_fold, pair_window = future_to_pair[future]
                try:
                    fold_id, window, row_indices, values = future.result()
                    if row_indices and len(values) == len(row_indices):
                        generated_policy_by_window[int(window)].loc[
                            row_indices, args.query_column
                        ] = values
                except Exception as exc:
                    generation_failed_pairs += 1
                    fold_id, window = int(pair_fold), int(pair_window)
                    print(
                        "[GEN][PAIR_ERROR] "
                        f"fold={fold_id} window={window} "
                        f"error={clean_text(str(exc)) or type(exc).__name__}",
                        flush=True,
                    )

                completed_generation_pairs += 1
                generation_left = total_generation_pairs - completed_generation_pairs
                print(
                    f"[GEN] {fold_id}, {window}, {generation_left}/{total_generation_pairs}",
                    flush=True,
                )
        if verbose_progress:
            print(
                "[GEN][PHASE_DONE] "
                f"completed={completed_generation_pairs}/{total_generation_pairs} "
                f"failed={generation_failed_pairs}",
                flush=True,
            )

    for window in windows:
        generated_df = generated_policy_by_window[int(window)]
        if args.generated_output is None:
            if len(windows) == 1:
                generated_output = args.output_dir / "generated_rag_summaries.csv"
            else:
                generated_output = (
                    args.output_dir
                    / "generated_rag_summaries_by_window"
                    / f"window_{window}.csv"
                )
        else:
            if len(windows) == 1:
                generated_output = args.generated_output
            else:
                suffix = args.generated_output.suffix or ".csv"
                stem = (
                    args.generated_output.stem
                    if args.generated_output.suffix
                    else args.generated_output.name
                )
                generated_output = (
                    args.generated_output.parent / f"{stem}_window_{window}{suffix}"
                )

        generated_output.parent.mkdir(parents=True, exist_ok=True)
        generated_df.to_csv(generated_output, index=False)
        generated_outputs_by_window[str(int(window))] = str(generated_output)

    all_results: list[dict] = []
    debug_fold0_doc_rows: list[dict[str, object]] = []

    def run_evaluation_pair(
        fold_id: int, window: int
    ) -> tuple[int, int, dict | None, list[dict[str, object]], bool]:
        if verbose_progress:
            print(
                f"[EVAL][PAIR_START] fold={fold_id} window={window}",
                flush=True,
            )
        active_policy_df = generated_policy_by_window[int(window)]
        (
            chunk_texts,
            chunk_doc_ids,
            chunk_geographies,
            chunk_sectors,
            chunk_ids,
            chunk_sources,
            chunk_embeddings,
        ) = load_fold_chunk_corpus(
            chroma_dir=args.chroma_dir,
            collection_name=args.chroma_collection,
            fold_id=fold_id,
        )

        rag_semantic = OpenAIChunkSemanticRetriever(
            chunk_texts=chunk_texts,
            chunk_doc_ids=chunk_doc_ids,
            chunk_geographies=chunk_geographies,
            chunk_sectors=chunk_sectors,
            chunk_ids=chunk_ids,
            chunk_sources=chunk_sources,
            model_name=args.encoder,
            use_metadata_filter=args.eval_use_metadata_filter,
            precomputed_chunk_embeddings=chunk_embeddings,
        )

        window_records = load_window_test_records(
            model_input_dir=args.model_input_dir,
            fold_id=fold_id,
            window=window,
        )

        debug_rows: list[dict[str, object]] = []
        if debug_mode and int(fold_id) == 0:
            debug_rows = collect_window_test_doc_ids(
                window_records=window_records,
                fold_id=fold_id,
                window=window,
            )

        test_df = build_window_test_dataframe(
            policy_df=active_policy_df,
            fold_id=fold_id,
            window_records=window_records,
            fold_col=args.fold_column,
            doc_id_col=args.doc_id_column,
            query_col=args.query_column,
        )

        if debug_mode:
            test_df = test_df.head(debug_max_queries).reset_index(drop=True)
        if test_df.empty:
            return int(fold_id), int(window), None, debug_rows, True

        method_name = "rag_summary_semantic"
        run_dir = args.output_dir / f"fold_{fold_id}" / f"window_{window}" / method_name
        result = evaluate_fold(
            fold_id=fold_id,
            window=window,
            method_name=method_name,
            fold_df=test_df,
            query_col=args.query_column,
            doc_id_col=args.doc_id_column,
            retriever=rag_semantic,
            k_values=k_values,
            save_retrieval_traces=args.save_retrieval_traces,
            retrieval_trace_top_k=args.retrieval_trace_top_k,
            retrieval_trace_output_path=(run_dir / "retrieval_traces.csv"),
        )
        result["fold"] = int(fold_id)
        result["window"] = int(window)
        result["method"] = method_name
        return int(fold_id), int(window), result, debug_rows, False

    eval_pairs = [
        (int(fold_id), int(window)) for fold_id in fold_ids for window in windows
    ]
    total_fold_window_pairs = len(eval_pairs)
    completed_fold_window_pairs = 0
    evaluation_workers = max(
        1,
        min(
            total_fold_window_pairs,
            int(args.fold_parallelism) * int(args.window_parallelism),
        ),
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=evaluation_workers
    ) as executor:
        if verbose_progress:
            print(
                f"[EVAL][PHASE_START] pairs={total_fold_window_pairs} workers={evaluation_workers}",
                flush=True,
            )
        future_to_pair = {
            executor.submit(run_evaluation_pair, fold_id, window): (fold_id, window)
            for fold_id, window in eval_pairs
        }
        evaluation_failed_pairs = 0
        pending_evaluation_futures = set(future_to_pair.keys())
        while pending_evaluation_futures:
            done_evaluation_futures, pending_evaluation_futures = (
                concurrent.futures.wait(
                    pending_evaluation_futures,
                    timeout=30,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
            )
            if not done_evaluation_futures:
                if verbose_progress:
                    sample_pending = [
                        future_to_pair[future]
                        for future in list(pending_evaluation_futures)[:5]
                    ]
                    print(
                        "[EVAL][HEARTBEAT] "
                        f"completed={completed_fold_window_pairs}/{total_fold_window_pairs} "
                        f"pending={len(pending_evaluation_futures)} "
                        f"sample_pending={sample_pending}",
                        flush=True,
                    )
                continue

            for future in done_evaluation_futures:
                pair_fold, pair_window = future_to_pair[future]
                try:
                    fold_id, window, result, debug_rows, is_empty = future.result()
                    if debug_rows:
                        debug_fold0_doc_rows.extend(debug_rows)

                    if is_empty:
                        print(
                            f"[WARN] Skip fold={fold_id}, window={window}: empty test dataframe"
                        )
                    elif result is not None:
                        all_results.append(result)
                        print(
                            f"Finished fold={fold_id}, window={window}, method=rag_summary_semantic, "
                            f"hit@5={result.get('hit@5', 0.0):.4f}, "
                            f"ndcg@5={result.get('ndcg@5', 0.0):.4f}, "
                            f"precision@5={result.get('precision@5', 0.0):.4f}, "
                            f"mrr@5={result.get('mrr@5', 0.0):.4f}"
                        )
                except Exception as exc:
                    evaluation_failed_pairs += 1
                    fold_id, window = int(pair_fold), int(pair_window)
                    print(
                        "[EVAL][PAIR_ERROR] "
                        f"fold={fold_id} window={window} "
                        f"error={clean_text(str(exc)) or type(exc).__name__}",
                        flush=True,
                    )

                completed_fold_window_pairs += 1
                left = total_fold_window_pairs - completed_fold_window_pairs
                print(
                    f"[EVAL] {fold_id}, {window}, {left}/{total_fold_window_pairs}",
                    flush=True,
                )
        if verbose_progress:
            print(
                "[EVAL][PHASE_DONE] "
                f"completed={completed_fold_window_pairs}/{total_fold_window_pairs} "
                f"failed={evaluation_failed_pairs}",
                flush=True,
            )

    if debug_mode and debug_fold0_doc_rows:
        debug_doc_df = pd.DataFrame(debug_fold0_doc_rows).drop_duplicates()
        debug_doc_df = debug_doc_df.sort_values(["window", "doc_id"]).reset_index(
            drop=True
        )
        debug_doc_path = args.output_dir / "debug_fold_0_test_doc_ids.csv"
        debug_doc_df.to_csv(debug_doc_path, index=False)
        print(f"Saved debug fold 0 doc IDs: {debug_doc_path}")

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df = results_df.sort_values(["fold", "window", "method"])
    results_path = args.output_dir / "all_fold_results.csv"
    results_df.to_csv(results_path, index=False)

    if not results_df.empty:
        summarize_fold_metrics(results_df, args.output_dir)

    metadata = {
        "task": "seekpolicy_rag_summary_experiments",
        "policy_input": str(args.policy_input),
        "model_input_dir": str(args.model_input_dir),
        "fold_column": args.fold_column,
        "doc_id_column": args.doc_id_column,
        "query_column": args.query_column,
        "generated_rag_summary": True,
        "generated_outputs_by_window": generated_outputs_by_window,
        "pool_mode": "same_fold_generation_pool_and_test_pool_eval",
        "chroma_dir": str(args.chroma_dir),
        "chroma_collection": args.chroma_collection,
        "openai_model": args.openai_model,
        "folds": fold_ids,
        "windows": windows,
        "encoder": args.encoder,
        "k_values": k_values,
        "save_retrieval_traces": bool(args.save_retrieval_traces),
        "retrieval_trace_top_k": int(args.retrieval_trace_top_k),
        "methods": [
            "rag_summary_semantic",
        ],
        "metrics": ["Hit@k", "Precision@k", "NDCG@k", "MRR@k"],
        "results_csv": str(results_path),
    }

    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved results: {results_path}")


if __name__ == "__main__":
    main()
