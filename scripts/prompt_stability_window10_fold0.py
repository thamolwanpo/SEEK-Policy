"""
Prompt Stability & Hyperparameter Analysis
==========================================
For each seed:
  1. Re-generates RAG summary queries for the fixed 50 test documents
     (window=10, fold=0) using the FULL seekpolicy generation pipeline
     (table selection → dataset analysis → chunk blend → final summary).
  2. Embeds those freshly generated queries and retrieves from the Chroma
     chunk corpus for the same fold.
  3. Reports Hit@k, Precision@k, NDCG@k, MRR@k per seed and stability
     statistics (mean / std / CV%) across seeds.
  4. Prints an Appendix table of every threshold / hyperparameter used.

Seeds run in parallel.  Each seed uses temperature=1 so the LLM produces
different outputs per run (temperature=0 would give identical results and
zero variance by design).  Pass --generation-temperature 0 to confirm that
and use it as a determinism sanity check.

Usage
-----
python stability_experiment.py \\
    --n-samples 50 \\
    --seeds 42,123,777 \\
    --window 10 \\
    --fold 0 \\
    --openai-model gpt-4o-mini \\
    --generation-temperature 1

Required env var: OPENAI_API_KEY (or set in .env)
"""

import argparse
import concurrent.futures
import json
import os
import random
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
except ImportError as exc:
    raise ImportError(
        "Missing dependency `langchain-openai`. "
        "Install with `pip install langchain-openai`."
    ) from exc

# ──────────────────────────────────────────────────────────────────────────────
# Constants (mirror main pipeline defaults)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_POLICY_INPUT = Path("data/csv/group_kfold_assignments.csv")
DEFAULT_MODEL_INPUT_DIR = Path("data/model_input/kfold")
DEFAULT_OUTPUT_DIR = Path("results/stability_experiments")
DEFAULT_CHROMA_DIR = Path("data/vectorstore/policy_chunks_chroma")
DEFAULT_CHROMA_COLLECTION = "policy_chunks_openai"
DEFAULT_OWID_DIR = Path("data/owid")
DEFAULT_ENCODER = "text-embedding-3-small"
DEFAULT_QUERY_COLUMN = "RAG_v1_summary"
DEFAULT_DOC_ID_COLUMN = "Document ID"
DEFAULT_FOLD_COLUMN = "fold"
DEFAULT_RAG_RETRIEVER_TOP_K = 1000

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI concurrency guard
# ──────────────────────────────────────────────────────────────────────────────
_OPENAI_SEMAPHORE: threading.BoundedSemaphore | None = None


def configure_openai_concurrency(limit: int) -> None:
    global _OPENAI_SEMAPHORE
    _OPENAI_SEMAPHORE = threading.BoundedSemaphore(max(int(limit), 1))


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(t in text for t in ("rate limit", "too many requests", "429", "quota"))


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
        sem = _OPENAI_SEMAPHORE
        if sem is not None:
            if not sem.acquire(timeout=300):
                raise RuntimeError("OpenAI semaphore acquire timed out after 300 s")
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt >= attempts - 1 or not _is_rate_limit_error(exc):
                raise
        finally:
            if sem is not None:
                sem.release()
        delay = min(delay * 2 if delay > 0 else 1.0, 16.0)
        time.sleep(delay)


# ──────────────────────────────────────────────────────────────────────────────
# Text / parsing helpers
# ──────────────────────────────────────────────────────────────────────────────


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def parse_csv_ints(raw: str) -> list[int]:
    vals = sorted({int(t.strip()) for t in raw.split(",") if t.strip()})
    if not vals:
        raise ValueError("Expected at least one integer")
    return vals


def ensure_openai_api_key_loaded() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("OPENAI_API_KEY is not set. Add it to `.env`.")


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────


def load_policy_df(
    path: Path, fold_col: str, doc_id_col: str, query_col: str
) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in (fold_col, doc_id_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {path}")
    df = df.copy()
    df[doc_id_col] = df[doc_id_col].astype(str)
    df[fold_col] = df[fold_col].astype(int)
    if query_col not in df.columns:
        df[query_col] = ""
    df[query_col] = df[query_col].apply(clean_text)
    return df[df[doc_id_col] != ""].reset_index(drop=True)


def load_window_test_records(
    model_input_dir: Path, fold_id: int, window: int
) -> list[dict]:
    path = model_input_dir / f"fold_{fold_id}" / f"window_{window}" / "test.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing test file: {path}. Run data-preparation scripts first."
        )
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    return payload


def metadata_get_str(metadata: dict, keys: list[str]) -> str:
    for k in keys:
        v = metadata.get(k)
        if v is not None:
            t = str(v).strip()
            if t:
                return t
    return ""


def metadata_get_int(metadata: dict, keys: list[str]) -> int | None:
    for k in keys:
        v = metadata.get(k)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return None


def extract_policy_year(row: pd.Series) -> int | None:
    raw = clean_text(row.get("Last event in timeline", ""))
    if not raw:
        return None
    try:
        return int(raw.split("-")[0])
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# OWID table catalog
# ──────────────────────────────────────────────────────────────────────────────


def discover_owid_table_catalog(
    owid_dir: Path,
) -> tuple[dict[str, Path], list[str]]:
    table_paths: dict[str, Path] = {}
    table_titles: list[str] = []
    if not owid_dir.exists():
        return table_paths, table_titles
    for file_path in sorted(owid_dir.rglob("*.meta.json")):
        try:
            with file_path.open("r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except Exception:
            continue
        title = ""
        if isinstance(obj, dict):
            raw = obj.get("title", None)
            if isinstance(raw, str) and raw.strip():
                title = raw.strip().lower()
            else:
                ds = obj.get("dataset", {})
                if isinstance(ds, dict):
                    dst = ds.get("title", None)
                    if isinstance(dst, str) and dst.strip():
                        title = dst.strip().lower()
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
            candidates = [clean_text(str(i)).lower() for i in parsed]
    except Exception:
        pass
    if not candidates:
        norm = (
            response_text.replace("\n", ",")
            .replace("•", ",")
            .replace(" - ", ",")
            .replace(";", ",")
        )
        candidates = [t.strip().lower() for t in norm.split(",") if t.strip()]
    low = response_text.lower()
    for title in known_titles:
        if title in low:
            candidates.append(title)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        item_c = clean_text(item).lower()
        if item_c in known_titles and item_c not in seen:
            deduped.append(item_c)
            seen.add(item_c)
            continue
        for title in known_titles:
            if item_c and item_c in title and title not in seen:
                deduped.append(title)
                seen.add(title)
    return deduped


# ──────────────────────────────────────────────────────────────────────────────
# Chunk corpus loader from Chroma
# ──────────────────────────────────────────────────────────────────────────────


def load_fold_chunk_corpus(
    chroma_dir: Path,
    collection_name: str,
    fold_id: int,
) -> tuple[
    list[str], list[str], list[str], list[str], list[str], list[str], np.ndarray | None
]:
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    if not chroma_dir.exists():
        raise FileNotFoundError(f"Chroma directory not found: {chroma_dir}")

    store = Chroma(
        collection_name=collection_name,
        embedding_function=None,
        persist_directory=str(chroma_dir),
    )

    page_size = 1000
    offset = 0
    texts, doc_ids, geos, sectors, chunk_ids, sources = [], [], [], [], [], []
    raw_embeddings: list[object] = []

    while True:
        payload = store.get(
            include=["documents", "metadatas", "embeddings"],
            limit=page_size,
            offset=offset,
        )
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []
        raw_emb = payload.get("embeddings")
        if isinstance(raw_emb, np.ndarray):
            embeddings = raw_emb.tolist()
        elif raw_emb is None:
            embeddings = []
        else:
            embeddings = list(raw_emb)
        if len(embeddings) != len(documents):
            embeddings = [None] * len(documents)

        if not documents:
            break

        for text, meta, emb in zip(documents, metadatas, embeddings):
            meta = meta or {}
            if metadata_get_int(meta, ["fold", "Fold"]) != int(fold_id):
                continue
            doc_id = metadata_get_str(
                meta, ["document_id", "Document ID", "doc_id", "documentId"]
            )
            cleaned = clean_text(text)
            if not doc_id or not cleaned:
                continue
            texts.append(cleaned)
            doc_ids.append(doc_id)
            geos.append(
                metadata_get_str(
                    meta,
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
            sectors.append(
                metadata_get_str(meta, ["sector", "Sector", "topic", "Topic"])
            )
            chunk_ids.append(metadata_get_str(meta, ["chunk_id", "chunkId"]))
            sources.append(metadata_get_str(meta, ["source", "file_path", "path"]))
            raw_embeddings.append(emb)

        if len(documents) < page_size:
            break
        offset += page_size

    if not texts:
        raise RuntimeError(
            f"No chunks found for fold={fold_id} in collection '{collection_name}' at {chroma_dir}."
        )

    precomputed: np.ndarray | None = None
    if raw_embeddings and all(
        isinstance(e, (list, tuple)) and len(e) > 0 for e in raw_embeddings
    ):
        try:
            precomputed = np.asarray(raw_embeddings, dtype=np.float64)
        except Exception:
            precomputed = None

    return texts, doc_ids, geos, sectors, chunk_ids, sources, precomputed


def load_existing_chroma_store(chroma_dir: Path, collection_name: str) -> object:
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings as _OAIEmb

    if not chroma_dir.exists():
        raise FileNotFoundError(f"Chroma directory not found: {chroma_dir}")
    return Chroma(
        collection_name=collection_name,
        embedding_function=_OAIEmb(),
        persist_directory=str(chroma_dir),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval helpers (identical to main pipeline)
# ──────────────────────────────────────────────────────────────────────────────


def aggregate_by_doc_max(
    scores: np.ndarray, doc_ids: list[str]
) -> tuple[np.ndarray, list[str]]:
    score_by_doc: dict[str, float] = {}
    for s, d in zip(scores.tolist(), doc_ids):
        score_by_doc[d] = max(score_by_doc.get(d, float("-inf")), float(s))
    uids = list(score_by_doc)
    return np.array([score_by_doc[d] for d in uids], dtype=np.float64), uids


def rank_from_scores(scores: np.ndarray, doc_ids: list[str]) -> list[str]:
    agg_scores, agg_ids = aggregate_by_doc_max(scores, doc_ids)
    if not agg_ids:
        return []
    return [agg_ids[i] for i in np.argsort(-agg_scores)]


def retrieve_top_chunks_for_query(
    query: str,
    country: str,
    sector: str,
    vector_store: object,
    allowed_doc_ids: set[str],
    top_docs: int = 3,
) -> list[str]:
    candidate_filters = []
    if country and sector:
        candidate_filters.append({"geography": country, "sector": sector})
    if country:
        candidate_filters.append({"geography": country})
    candidate_filters.append({})

    docs: list = []
    for fq in candidate_filters:
        try:
            docs = call_openai_with_guard(
                vector_store.similarity_search_with_score,
                query,
                k=DEFAULT_RAG_RETRIEVER_TOP_K,
                filter=fq or None,
            )
        except Exception:
            docs = []
        if docs:
            break

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
        chunks.append(chunk)
        if doc_id:
            seen_doc_ids.add(doc_id)
        if len(seen_doc_ids) >= top_docs:
            break
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────


def hit_at_k(ranked: list[str], target: str, k: int) -> float:
    return 1.0 if target in ranked[:k] else 0.0


def precision_at_k(ranked: list[str], target: str, k: int) -> float:
    return (1.0 / k) if target in ranked[:k] else 0.0


def ndcg_at_k(ranked: list[str], target: str, k: int) -> float:
    if target not in ranked[:k]:
        return 0.0
    return 1.0 / np.log2(ranked[:k].index(target) + 2)


def mrr_at_k(ranked: list[str], target: str, k: int) -> float:
    if target not in ranked[:k]:
        return 0.0
    return 1.0 / (ranked[:k].index(target) + 1)


# ──────────────────────────────────────────────────────────────────────────────
# Chunk retriever for evaluation (uses pre-computed Chroma embeddings)
# ──────────────────────────────────────────────────────────────────────────────


class EvalChunkRetriever:
    """Embeds a fresh query string and scores against the pre-loaded corpus."""

    def __init__(
        self,
        chunk_texts: list[str],
        chunk_doc_ids: list[str],
        chunk_geographies: list[str],
        chunk_sectors: list[str],
        model_name: str,
        use_metadata_filter: bool = False,
        precomputed_chunk_embeddings: np.ndarray | None = None,
    ):
        ensure_openai_api_key_loaded()
        self.embeddings_model = OpenAIEmbeddings(model=model_name)
        self.chunk_texts = chunk_texts
        self.chunk_doc_ids = chunk_doc_ids
        self.chunk_geographies = [clean_text(g) for g in chunk_geographies]
        self.chunk_sectors = [clean_text(s) for s in chunk_sectors]
        self.use_metadata_filter = bool(use_metadata_filter)

        if precomputed_chunk_embeddings is not None:
            self.chunk_embeddings = precomputed_chunk_embeddings.astype(np.float64)
        else:
            vectors: list[list[float]] = []
            batch_size = 128
            for start in range(0, len(chunk_texts), batch_size):
                batch = chunk_texts[start : start + batch_size]
                vectors.extend(
                    call_openai_with_guard(self.embeddings_model.embed_documents, batch)
                )
            self.chunk_embeddings = np.asarray(vectors, dtype=np.float64)

    def _candidate_indices(self, country: str, sector: str) -> np.ndarray:
        if not self.use_metadata_filter:
            return np.arange(len(self.chunk_texts), dtype=np.int64)

        def select(fc: str, fs: str) -> np.ndarray:
            return np.array(
                [
                    i
                    for i, (g, s) in enumerate(
                        zip(self.chunk_geographies, self.chunk_sectors)
                    )
                    if (not fc or g == fc) and (not fs or s == fs)
                ],
                dtype=np.int64,
            )

        for fc, fs in [(country, sector), (country, ""), ("", "")]:
            trial = select(fc, fs)
            if len(trial):
                return trial
        return np.arange(len(self.chunk_texts), dtype=np.int64)

    def retrieve(
        self, query: str, k: int, country: str = "", sector: str = ""
    ) -> list[str]:
        if not self.chunk_texts:
            return []
        cand = self._candidate_indices(clean_text(country), clean_text(sector))
        if not len(cand):
            return []
        q_emb = np.asarray(
            [call_openai_with_guard(self.embeddings_model.embed_query, query)],
            dtype=np.float64,
        )
        scores = cosine_similarity(q_emb, self.chunk_embeddings[cand])[0].astype(
            np.float64
        )
        ranked = rank_from_scores(scores, [self.chunk_doc_ids[i] for i in cand])
        return ranked[:k]


# ──────────────────────────────────────────────────────────────────────────────
# Full seekpolicy RAG generation for one document row
# ──────────────────────────────────────────────────────────────────────────────


def generate_rag_query_for_doc(
    row: pd.Series,
    doc_id_col: str,
    vector_store: object,
    allowed_doc_ids: set[str],
    table_names: dict[str, Path],
    table_titles: list[str],
    llm: object,
    history_window: int,
) -> str:
    """Runs the full seekpolicy generation pipeline for one document.
    Returns the generated RAG summary string, or '' on failure.
    """
    sector = clean_text(row.get("Sector", ""))
    instrument = clean_text(row.get("Instrument", ""))
    keyword = clean_text(row.get("Keyword", ""))
    topic = clean_text(row.get("Topic/Response", ""))
    hazard = clean_text(row.get("Hazard", ""))
    country = clean_text(row.get("Geography", ""))
    region = clean_text(row.get("region", ""))
    policy_year = extract_policy_year(row)

    policy_metadata = (
        f"Targeted Sector: {sector}\n"
        f"Policy Instrument: {instrument}\n"
        f"Keywords: {keyword}\n"
        f"Topics: {topic}\n"
        f"Hazards: {hazard}"
    )
    if not policy_metadata.strip():
        return ""

    known_titles = set(table_titles)

    # ── Step 1: table selection ──────────────────────────────────────────
    selected_tables_prompt = (
        f"Based on this table's names: {table_titles}, I want you to select all the "
        f"table's names from the list that might be directly related to the creation "
        f"of the policy with the following metadata.\n\n"
        f"POLICY METADATA:\n{policy_metadata}\n\n"
        f"Return in list with no explaination.\nBegin:\n"
    )
    try:
        resp = call_openai_with_guard(llm.invoke, selected_tables_prompt)
        selected_titles = parse_selected_table_titles(str(resp.content), known_titles)
    except Exception:
        return ""

    # ── Step 2: per-table dataset analysis ──────────────────────────────
    def retrieve_window_data(frame: pd.DataFrame, location: str, test_year: int) -> str:
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
            return filtered.to_string(index=False, float_format=lambda v: f"{v:.2f}")
        except Exception:
            return filtered.to_string(index=False)

    def analyse_one_title(title: str) -> str | None:
        fp = table_names.get(title)
        if fp is None:
            return None
        try:
            with fp.open("r", encoding="utf-8") as fh:
                meta_obj = json.load(fh)
        except Exception:
            return None
        dataset = meta_obj if isinstance(meta_obj, dict) else {}
        desc = None
        if "dataset" in dataset and isinstance(dataset["dataset"], dict):
            desc = dataset["dataset"].get("description")
        elif "description" in dataset:
            desc = dataset.get("description")
        text = f"DATASET:\nTitle: {title}\n"
        if desc:
            text += f"Description: {desc}\n"
        csv_path = Path(str(fp).replace(".meta.json", ".csv"))
        if not csv_path.exists():
            return None
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            return None
        if country and policy_year is not None:
            text += f"Focused Country Data: {country}\n"
            text += retrieve_window_data(frame, country, policy_year)
        if region and policy_year is not None:
            text += f"\n{region} Data:\n"
            text += retrieve_window_data(frame, region, policy_year)
        if policy_year is not None:
            text += "\nWorld Data:\n"
            text += retrieve_window_data(frame, "World", policy_year)
        if not text.strip():
            return None
        try:
            prompt = (
                f"You are policy analyzer expert.\n"
                f"You are to analyse the following data, but keep in mind that you are "
                f"focused at {sector} aspects.\n"
                f"Here is the data you will be analysed.\n\n{text}\n\n----\n\n"
                f"Output only one paragraph with no explanation.\nBegin:\n"
            )
            r = call_openai_with_guard(llm.invoke, prompt)
            result = clean_text(r.content)
            return result if result else None
        except Exception:
            return None

    step_1_queries: list[str] = []
    for title in selected_titles:
        result = analyse_one_title(title)
        if result:
            step_1_queries.append(result)

    if not step_1_queries:
        return ""

    # ── Step 3: blend each analysis with related chunks ──────────────────
    def blend_one_query(step_query: str) -> str | None:
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
            "Generate a single coherent paragraph based on the following time series "
            "analysis and related content. "
            "Ensure that the writing style of the generated paragraph is a blend of the "
            "three other chunks, while incorporating the insights from the time series "
            "analysis. "
            "Do not mention the name of any policy, plan, act, or refer to "
            "'time series analysis' explicitly."
        )
        input_text = (
            instruction
            + "\n\nTime Series Analysis Chunk:\n"
            + step_query
            + "\n\nOther Related Chunks:\n"
            + "\n".join(related_chunks)
        )
        try:
            r = call_openai_with_guard(llm.invoke, input_text)
            result = clean_text(r.content)
            return result if result else None
        except Exception:
            return None

    all_new_chunks: list[str] = []
    for sq in step_1_queries:
        result = blend_one_query(sq)
        if result:
            all_new_chunks.append(result)

    unique_chunks = list(dict.fromkeys(all_new_chunks))
    if not unique_chunks:
        return ""

    # ── Step 4: final summary ────────────────────────────────────────────
    final_prompt = (
        "You are policy analyzer expert.\n"
        "You are to summarize these chunks of information to one paragraph:\n\n"
        f"{'[NEW_CHUNK]'.join(unique_chunks)}\n\n----\n\n"
        "Output only one paragraph with no explanation.\nBegin:\n"
    )
    try:
        r = call_openai_with_guard(llm.invoke, final_prompt)
        return clean_text(r.content)
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Per-seed worker: generate then evaluate
# ──────────────────────────────────────────────────────────────────────────────


def run_seed(
    seed: int,
    sample_df: pd.DataFrame,
    doc_id_col: str,
    query_col: str,
    k_values: list[int],
    # generation
    chroma_dir: Path,
    chroma_collection: str,
    allowed_doc_ids: set[str],
    table_names: dict[str, Path],
    table_titles: list[str],
    openai_model: str,
    generation_temperature: float,
    history_window: int,
    # evaluation
    chunk_texts: list[str],
    chunk_doc_ids: list[str],
    chunk_geographies: list[str],
    chunk_sectors: list[str],
    precomputed_embeddings: np.ndarray | None,
    encoder: str,
    use_metadata_filter: bool,
    output_dir: Path,
) -> dict[str, Any]:
    from langchain_openai import ChatOpenAI

    print(f"[SEED={seed}] Starting generation for {len(sample_df)} docs …", flush=True)

    # Each seed gets its own Chroma handle and LLM instance with its own seed
    vector_store = load_existing_chroma_store(chroma_dir, chroma_collection)
    llm = ChatOpenAI(
        model=openai_model,
        temperature=generation_temperature,
        seed=seed,  # makes each seed run reproducible yet different from others
    )

    # ── Generate a fresh RAG query for every sample doc ─────────────────
    generated_queries: dict[str, str] = {}
    for _, row in sample_df.iterrows():
        doc_id = clean_text(str(row.get(doc_id_col, "")))
        if not doc_id:
            continue
        print(f"[SEED={seed}] Generating for doc_id={doc_id} …", flush=True)
        query_text = generate_rag_query_for_doc(
            row=row,
            doc_id_col=doc_id_col,
            vector_store=vector_store,
            allowed_doc_ids=allowed_doc_ids,
            table_names=table_names,
            table_titles=table_titles,
            llm=llm,
            history_window=history_window,
        )
        generated_queries[doc_id] = query_text
        status = f"✓ {len(query_text)} chars" if query_text else "✗ empty/failed"
        print(f"[SEED={seed}]   {status}", flush=True)

    # Save generated queries for manual inspection
    gen_rows = [
        {doc_id_col: did, query_col: txt} for did, txt in generated_queries.items()
    ]
    gen_df = pd.DataFrame(gen_rows)
    gen_path = output_dir / f"generated_queries_seed_{seed}.csv"
    gen_df.to_csv(gen_path, index=False)
    n_filled = sum(1 for t in generated_queries.values() if t)
    print(
        f"[SEED={seed}] Generation done. filled={n_filled}/{len(sample_df)} → {gen_path}",
        flush=True,
    )

    # ── Build retriever (chunk embeddings are pre-loaded — no re-embedding) ─
    print(f"[SEED={seed}] Building retriever …", flush=True)
    retriever = EvalChunkRetriever(
        chunk_texts=chunk_texts,
        chunk_doc_ids=chunk_doc_ids,
        chunk_geographies=chunk_geographies,
        chunk_sectors=chunk_sectors,
        model_name=encoder,
        use_metadata_filter=use_metadata_filter,
        precomputed_chunk_embeddings=precomputed_embeddings,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────
    print(f"[SEED={seed}] Evaluating …", flush=True)
    records: dict[str, list[float]] = {
        **{f"hit@{k}": [] for k in k_values},
        **{f"precision@{k}": [] for k in k_values},
        **{f"ndcg@{k}": [] for k in k_values},
        **{f"mrr@{k}": [] for k in k_values},
    }
    max_k = max(k_values)
    skipped = 0

    for _, row in sample_df.iterrows():
        doc_id = clean_text(str(row.get(doc_id_col, "")))
        target = doc_id
        country = clean_text(row.get("Geography", ""))
        sector = clean_text(row.get("Sector", ""))
        query = generated_queries.get(doc_id, "")

        if not query or not target:
            skipped += 1
            continue

        ranked = retriever.retrieve(
            query=query, k=max_k, country=country, sector=sector
        )
        if not ranked:
            skipped += 1
            continue

        for k in k_values:
            records[f"hit@{k}"].append(hit_at_k(ranked, target, k))
            records[f"precision@{k}"].append(precision_at_k(ranked, target, k))
            records[f"ndcg@{k}"].append(ndcg_at_k(ranked, target, k))
            records[f"mrr@{k}"].append(mrr_at_k(ranked, target, k))

    metrics: dict[str, Any] = {
        name: float(np.mean(vals)) if vals else 0.0 for name, vals in records.items()
    }
    metrics["seed"] = seed
    metrics["queries_evaluated"] = len(sample_df) - skipped
    metrics["queries_total"] = len(sample_df)
    metrics["queries_skipped"] = skipped

    print(
        f"[SEED={seed}] Eval done. evaluated={metrics['queries_evaluated']} "
        f"hit@5={metrics.get('hit@5', 0):.4f} ndcg@5={metrics.get('ndcg@5', 0):.4f}",
        flush=True,
    )
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter catalogue (Appendix)
# ──────────────────────────────────────────────────────────────────────────────

HYPERPARAMETER_TABLE = [
    (
        "Data / Missing Values",
        "Rows with empty Document ID",
        "Dropped",
        "load_policy_df()",
    ),
    (
        "Data / Missing Values",
        "Rows with empty generated query",
        "Skipped at eval",
        "run_seed()",
    ),
    (
        "Data / Text normalisation",
        "Whitespace normalisation",
        "Collapsed to single space",
        "clean_text()",
    ),
    (
        "Retrieval / Embedding",
        "Embedding model",
        "text-embedding-3-small (default)",
        "--encoder",
    ),
    (
        "Retrieval / Embedding",
        "Embedding batch size",
        "128 chunks per API call",
        "EvalChunkRetriever",
    ),
    (
        "Retrieval / Scoring",
        "Similarity metric",
        "Cosine similarity",
        "sklearn cosine_similarity",
    ),
    (
        "Retrieval / Aggregation",
        "Multi-chunk → doc score",
        "Max-pooling over chunk scores",
        "aggregate_by_doc_max()",
    ),
    (
        "Retrieval / Filter",
        "Metadata filter (geo/sector)",
        "OFF by default; cascades country+sector → country → none",
        "--eval-use-metadata-filter",
    ),
    (
        "Retrieval / Chroma",
        "Top-k candidates from Chroma (generation)",
        "1 000",
        "DEFAULT_RAG_RETRIEVER_TOP_K",
    ),
    (
        "Retrieval / Chroma",
        "Top docs passed to blend prompt",
        "3",
        "retrieve_top_chunks_for_query()",
    ),
    ("Generation / LLM", "LLM model", "gpt-4o-mini (default)", "--openai-model"),
    (
        "Generation / LLM",
        "Temperature",
        "1.0 (stochastic, varies per seed)",
        "--generation-temperature",
    ),
    (
        "Generation / LLM",
        "Seed param passed to ChatOpenAI",
        "Same integer as stability seed",
        "ChatOpenAI(seed=seed)",
    ),
    (
        "Generation / History",
        "Time-series window (years before policy year)",
        "10 (default)",
        "--window",
    ),
    (
        "Generation / RAG k",
        "Chunks retrieved per blend step",
        "3",
        "retrieve_top_chunks_for_query()",
    ),
    (
        "CV / Splitting",
        "CV strategy",
        "Group k-fold (by document)",
        "group_kfold_assignments.csv",
    ),
    (
        "CV / Test isolation",
        "Corpus pool for evaluation",
        "Same fold as query document",
        "load_fold_chunk_corpus()",
    ),
    ("Metrics / Hit@k", "k values evaluated", "1, 5, 10 (default)", "--k-values"),
    (
        "Metrics / NDCG@k",
        "Relevance model",
        "Binary (1 target doc per query)",
        "ndcg_at_k()",
    ),
    (
        "Metrics / MRR@k",
        "Rank reciprocal",
        "1 / rank of first relevant doc",
        "mrr_at_k()",
    ),
    ("Stability / Seeds", "Number of repetitions", "3", "--seeds"),
    (
        "Stability / Variance src",
        "What changes between seeds",
        "LLM outputs (temperature=1) + ChatOpenAI seed",
        "--generation-temperature",
    ),
    (
        "Stability / Sample size",
        "Test queries per seed run",
        "50 (default)",
        "--n-samples",
    ),
    (
        "Stability / Sampling",
        "Sample selection",
        "Deterministic first-N from test.json",
        "fixed sample_df",
    ),
    (
        "API / Concurrency",
        "Global OpenAI semaphore limit",
        "2 (default)",
        "--openai-concurrency",
    ),
    ("API / Retry", "Max retries on rate-limit", "5", "call_openai_with_guard()"),
    (
        "API / Retry",
        "Base delay (exponential backoff)",
        "1 s, max 16 s",
        "call_openai_with_guard()",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prompt stability experiment: re-generate RAG queries per seed "
            "using the full seekpolicy pipeline, then evaluate retrieval metrics."
        )
    )
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--seeds", type=str, default="42,123,777")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--k-values", type=str, default="1,5,10")
    p.add_argument("--encoder", type=str, default=DEFAULT_ENCODER)
    p.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    p.add_argument(
        "--generation-temperature",
        type=float,
        default=1.0,
        help=(
            "LLM temperature for generation. 1.0 (default) = stochastic, different output per seed. "
            "0 = deterministic sanity-check (all seeds produce identical queries)."
        ),
    )
    p.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    p.add_argument("--chroma-collection", type=str, default=DEFAULT_CHROMA_COLLECTION)
    p.add_argument("--policy-input", type=Path, default=DEFAULT_POLICY_INPUT)
    p.add_argument("--model-input-dir", type=Path, default=DEFAULT_MODEL_INPUT_DIR)
    p.add_argument("--query-column", type=str, default=DEFAULT_QUERY_COLUMN)
    p.add_argument("--doc-id-column", type=str, default=DEFAULT_DOC_ID_COLUMN)
    p.add_argument("--fold-column", type=str, default=DEFAULT_FOLD_COLUMN)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--eval-use-metadata-filter", action="store_true")
    p.add_argument("--openai-concurrency", type=int, default=2)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_openai_api_key_loaded()
    configure_openai_concurrency(args.openai_concurrency)

    seeds = parse_csv_ints(args.seeds)
    k_values = parse_csv_ints(args.k_values)
    fold_id = args.fold
    window = args.window
    n_samples = max(1, args.n_samples)

    print(
        f"[STABILITY] seeds={seeds} window={window} fold={fold_id} "
        f"n_samples={n_samples} temperature={args.generation_temperature}",
        flush=True,
    )

    # ── 1. Load policy metadata ──────────────────────────────────────────
    policy_df = load_policy_df(
        path=args.policy_input,
        fold_col=args.fold_column,
        doc_id_col=args.doc_id_column,
        query_col=args.query_column,
    )

    # ── 2. Build fixed sample_df (same docs across all seeds) ────────────
    window_records = load_window_test_records(
        model_input_dir=args.model_input_dir,
        fold_id=fold_id,
        window=window,
    )
    print(
        f"[STABILITY] Total test records fold={fold_id} window={window}: {len(window_records)}",
        flush=True,
    )

    fold_policy = policy_df[policy_df[args.fold_column] == fold_id].copy()
    allowed_doc_ids = {
        clean_text(str(v))
        for v in fold_policy[args.doc_id_column].astype(str).tolist()
        if clean_text(str(v))
    }

    # Build full row metadata lookup (Geography, Sector, all other columns)
    doc_meta: dict[str, dict] = {}
    for _, row in fold_policy.iterrows():
        did = str(row.get(args.doc_id_column, "")).strip()
        if did and did not in doc_meta:
            doc_meta[did] = row.to_dict()

    rows_for_df = []
    for rec in window_records:
        did = clean_text(rec.get("doc_id", ""))
        if not did:
            continue
        meta = doc_meta.get(did, {})
        rows_for_df.append({**meta, args.doc_id_column: did})

    all_test_df = (
        pd.DataFrame(rows_for_df)
        .drop_duplicates(subset=[args.doc_id_column])
        .reset_index(drop=True)
    )
    sample_df = all_test_df.head(n_samples).reset_index(drop=True)
    print(f"[STABILITY] Fixed sample size: {len(sample_df)} rows", flush=True)
    sample_df.to_csv(args.output_dir / "fixed_sample.csv", index=False)

    # Sanity check: confirm key columns exist
    missing_cols = [c for c in ("Geography", "Sector") if c not in sample_df.columns]
    if missing_cols:
        print(
            f"[STABILITY][WARN] sample_df missing columns: {missing_cols} — geo/sector filtering disabled",
            flush=True,
        )

    # ── 3. Load OWID table catalog ────────────────────────────────────────
    table_names, table_titles = discover_owid_table_catalog(DEFAULT_OWID_DIR)
    print(f"[STABILITY] OWID tables found: {len(table_titles)}", flush=True)
    if not table_titles:
        print(
            f"[STABILITY][WARN] No OWID tables found at {DEFAULT_OWID_DIR}. "
            "Generation will produce empty queries for all docs.",
            flush=True,
        )

    # ── 4. Load chunk corpus once ─────────────────────────────────────────
    print(f"[STABILITY] Loading chunk corpus for fold={fold_id} …", flush=True)
    (
        chunk_texts,
        chunk_doc_ids,
        chunk_geographies,
        chunk_sectors,
        chunk_ids,
        chunk_sources,
        precomputed_embeddings,
    ) = load_fold_chunk_corpus(
        chroma_dir=args.chroma_dir,
        collection_name=args.chroma_collection,
        fold_id=fold_id,
    )
    print(f"[STABILITY] Loaded {len(chunk_texts)} chunks.", flush=True)

    # ── 5. Run seeds in parallel ──────────────────────────────────────────
    seed_results: list[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(seeds)) as executor:
        future_to_seed = {
            executor.submit(
                run_seed,
                seed,
                sample_df,
                args.doc_id_column,
                args.query_column,
                k_values,
                args.chroma_dir,
                args.chroma_collection,
                allowed_doc_ids,
                table_names,
                table_titles,
                args.openai_model,
                args.generation_temperature,
                window,
                chunk_texts,
                chunk_doc_ids,
                chunk_geographies,
                chunk_sectors,
                precomputed_embeddings,
                args.encoder,
                args.eval_use_metadata_filter,
                args.output_dir,
            ): seed
            for seed in seeds
        }
        for future in concurrent.futures.as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                result = future.result()
                seed_results.append(result)
            except Exception as exc:
                print(f"[SEED={seed}] FAILED: {exc}", flush=True)

    if not seed_results:
        print("[STABILITY] No seed results collected. Exiting.", flush=True)
        return

    seed_results.sort(key=lambda r: r["seed"])

    # ── 6. Stability statistics ───────────────────────────────────────────
    skip_cols = {"seed", "queries_evaluated", "queries_total", "queries_skipped"}
    metric_cols = [c for c in seed_results[0] if c not in skip_cols]

    stability_rows = []
    for metric in metric_cols:
        vals = np.array([r[metric] for r in seed_results], dtype=np.float64)
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv_v = (std_v / mean_v * 100) if mean_v > 0 else 0.0
        stability_rows.append(
            {
                "metric": metric,
                "mean": round(mean_v, 6),
                "std": round(std_v, 6),
                "cv_pct": round(cv_v, 2),
                "min": round(float(vals.min()), 6),
                "max": round(float(vals.max()), 6),
                **{f"seed_{r['seed']}": round(r[metric], 6) for r in seed_results},
            }
        )

    per_seed_df = pd.DataFrame(seed_results)
    stability_df = pd.DataFrame(stability_rows)
    hyperparam_df = pd.DataFrame(
        HYPERPARAMETER_TABLE,
        columns=["Category", "Hyperparameter", "Value", "Code Reference"],
    )

    per_seed_path = args.output_dir / "per_seed_metrics.csv"
    stability_path = args.output_dir / "stability_summary.csv"
    hyperparam_path = args.output_dir / "hyperparameter_table.csv"

    per_seed_df.to_csv(per_seed_path, index=False)
    stability_df.to_csv(stability_path, index=False)
    hyperparam_df.to_csv(hyperparam_path, index=False)

    # ── 7. Print report ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STABILITY REPORT")
    print(
        f"  window={window}  fold={fold_id}  n_samples={len(sample_df)}  seeds={seeds}"
    )
    print(f"  temperature={args.generation_temperature}  model={args.openai_model}")
    print("=" * 70)

    display_cols = ["seed", "queries_evaluated", "queries_skipped"] + [
        c for c in metric_cols if "@" in c
    ]
    print("\n── Per-Seed Metrics ──")
    print(per_seed_df[display_cols].to_string(index=False))

    print("\n── Stability Summary (mean ± std, CV%) ──")
    print(
        stability_df[["metric", "mean", "std", "cv_pct", "min", "max"]].to_string(
            index=False
        )
    )

    print("\n── APPENDIX: Hyperparameter Catalogue ──")
    print(hyperparam_df.to_string(index=False))
    print()

    # ── 8. Save run metadata ──────────────────────────────────────────────
    meta = {
        "experiment": "prompt_stability_with_regeneration",
        "fold": fold_id,
        "window": window,
        "n_samples": len(sample_df),
        "seeds": seeds,
        "k_values": k_values,
        "encoder": args.encoder,
        "openai_model": args.openai_model,
        "generation_temperature": args.generation_temperature,
        "eval_use_metadata_filter": args.eval_use_metadata_filter,
        "chunks_in_corpus": len(chunk_texts),
        "outputs": {
            "per_seed_metrics": str(per_seed_path),
            "stability_summary": str(stability_path),
            "hyperparameter_table": str(hyperparam_path),
            "fixed_sample": str(args.output_dir / "fixed_sample.csv"),
        },
    }
    with (args.output_dir / "stability_metadata.json").open("w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[STABILITY] Outputs written to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
