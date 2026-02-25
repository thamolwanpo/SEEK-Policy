import argparse
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
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Missing dependency `langchain-openai`. Install with `pip install langchain-openai`."
    ) from exc


DEFAULT_POLICY_INPUT = Path("data/csv/group_kfold_assignments.csv")
DEFAULT_OWID_DIR = Path("data/owid")
DEFAULT_OUTPUT_DIR = Path("results/sax_align_table_selector")
DEFAULT_OUTPUT_CSV = "selected_tables.csv"
DEFAULT_CACHE_FILE = "title_embeddings.npz"
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_TOP_N = 8
DEFAULT_MODEL_INPUT_DIR = Path("data/model_input/kfold")
DEFAULT_SELECTOR_WINDOW = 10
DEFAULT_FOLD_COLUMN = "fold"
DEFAULT_DOC_ID_COLUMN = "Document ID"
_OPENAI_SEMAPHORE: threading.BoundedSemaphore | None = None


def configure_openai_concurrency(limit: int) -> None:
    global _OPENAI_SEMAPHORE
    _OPENAI_SEMAPHORE = threading.BoundedSemaphore(max(int(limit), 1))


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
    delay = max(float(base_delay_seconds), 0.0)
    attempts = max(int(max_retries), 1)

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


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def ensure_openai_api_key_loaded() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("OPENAI_API_KEY is not set. Add it to `.env`.")


def discover_owid_titles(owid_dir: Path) -> list[str]:
    titles: list[str] = []
    if not owid_dir.exists():
        return titles

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

        if title:
            titles.append(title)

    return sorted(set(titles))


def load_policy_df(path: Path, fold_col: str, doc_id_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {fold_col, doc_id_col}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    out = df.copy()
    out[doc_id_col] = out[doc_id_col].astype(str).apply(clean_text)
    out[fold_col] = out[fold_col].astype(int)
    out = out[out[doc_id_col] != ""]
    return out.reset_index(drop=True)


def policy_metadata_text(row: pd.Series) -> str:
    sector = clean_text(row.get("Sector", ""))
    instrument = clean_text(row.get("Instrument", ""))
    keyword = clean_text(row.get("Keyword", ""))
    topic = clean_text(row.get("Topic/Response", ""))
    hazard = clean_text(row.get("Hazard", ""))

    lines = [
        f"Targeted Sector: {sector}",
        f"Policy Instrument: {instrument}",
        f"Keywords: {keyword}",
        f"Topics: {topic}",
        f"Hazards: {hazard}",
    ]
    return "\n".join(lines)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def embed_texts(
    embeddings: OpenAIEmbeddings,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float64)

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(call_openai_with_guard(embeddings.embed_documents, batch))
    return np.asarray(vectors, dtype=np.float64)


def load_title_cache(cache_path: Path) -> tuple[list[str], np.ndarray] | None:
    if not cache_path.exists():
        return None
    try:
        payload = np.load(cache_path, allow_pickle=False)
        titles = payload["titles"].tolist()
        vectors = payload["vectors"]
        if not isinstance(titles, list) or not isinstance(vectors, np.ndarray):
            return None
        return [str(item) for item in titles], np.asarray(vectors, dtype=np.float64)
    except Exception:
        return None


def save_title_cache(cache_path: Path, titles: list[str], vectors: np.ndarray) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, titles=np.asarray(titles), vectors=vectors)


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def resolve_output_path(
    output_dir: Path,
    output_file: str,
    selector_method: str,
    fold: int | None,
    run_all_methods: bool,
    prevent_overwrite: bool,
) -> Path:
    requested = Path(output_file)
    using_default_name = requested.name == DEFAULT_OUTPUT_CSV
    if using_default_name or run_all_methods:
        fold_tag = "all_folds" if fold is None else f"fold_{int(fold)}"
        base_name = f"selected_tables_{selector_method}_{fold_tag}.csv"
        path = output_dir / base_name
    else:
        path = output_dir / requested

    if prevent_overwrite:
        return ensure_unique_path(path)
    return path


def resolve_metadata_path(
    output_dir: Path,
    selector_method: str,
    run_all_methods: bool,
    prevent_overwrite: bool,
) -> Path:
    if run_all_methods:
        path = output_dir / f"run_metadata_{selector_method}.json"
    else:
        path = output_dir / "run_metadata.json"
    if prevent_overwrite:
        return ensure_unique_path(path)
    return path


def select_titles_for_policies(
    policy_df: pd.DataFrame,
    titles: list[str],
    title_vectors: np.ndarray,
    embeddings: OpenAIEmbeddings,
    top_n: int,
    batch_size: int,
    fold_col: str,
    doc_id_col: str,
) -> pd.DataFrame:
    metadata_texts = [policy_metadata_text(row) for _, row in policy_df.iterrows()]
    metadata_vectors = embed_texts(embeddings, metadata_texts, batch_size)

    if title_vectors.shape[0] == 0 or metadata_vectors.shape[0] == 0:
        return pd.DataFrame(
            columns=[
                fold_col,
                doc_id_col,
                "policy_metadata",
                "selected_titles",
                "selected_scores",
            ]
        )

    sims = cosine_similarity(metadata_vectors, title_vectors)

    rows: list[dict[str, object]] = []
    max_pick = max(int(top_n), 1)
    for idx in range(len(policy_df)):
        row = policy_df.iloc[idx]
        score_row = sims[idx]
        order = np.argsort(-score_row)
        pick = order[:max_pick]

        selected_titles = [titles[int(i)] for i in pick]
        selected_scores = [float(score_row[int(i)]) for i in pick]
        rows.append(
            {
                fold_col: int(row[fold_col]),
                doc_id_col: clean_text(row[doc_id_col]),
                "policy_metadata": metadata_texts[idx],
                "selected_titles": json.dumps(selected_titles, ensure_ascii=False),
                "selected_scores": json.dumps(selected_scores),
            }
        )

    return pd.DataFrame(rows)


def parse_selected_table_titles(
    response_text: str,
    known_titles: set[str],
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


def select_titles_for_policies_llm_prompt(
    policy_df: pd.DataFrame,
    titles: list[str],
    fold_col: str,
    doc_id_col: str,
    top_n: int,
    llm_model: str,
) -> pd.DataFrame:
    llm = ChatOpenAI(model=llm_model, temperature=0)
    known_titles = set(titles)
    max_pick = max(int(top_n), 1)

    rows: list[dict[str, object]] = []
    for _, row in policy_df.iterrows():
        metadata_text = policy_metadata_text(row)
        prompt = f"""
Based on this table's names: {titles}, I want you to select all the table's names from the list that might be directly related to the creation of the policy with the following metadata.

POLICY METADATA:
{metadata_text}

Return in list with no explaination.
Begin:
"""
        try:
            response = call_openai_with_guard(llm.invoke, prompt)
            selected_titles = parse_selected_table_titles(
                str(response.content), known_titles
            )[:max_pick]
        except Exception:
            selected_titles = []

        rows.append(
            {
                fold_col: int(row[fold_col]),
                doc_id_col: clean_text(row[doc_id_col]),
                "policy_metadata": metadata_text,
                "selected_titles": json.dumps(selected_titles, ensure_ascii=False),
                "selected_scores": json.dumps([]),
            }
        )

    return pd.DataFrame(rows)


def extract_policy_year(row: pd.Series) -> int | None:
    raw = clean_text(row.get("Last event in timeline", ""))
    if not raw:
        return None
    try:
        return int(raw.split("-")[0])
    except Exception:
        return None


def summarize_series(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return np.zeros(8, dtype=np.float64)

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    delta = float(arr[-1] - arr[0])
    x = np.arange(arr.size, dtype=np.float64)
    slope = float(np.polyfit(x, arr, 1)[0])
    scale = max(abs(mean_val), 1e-6)
    slope_norm = slope / scale
    delta_norm = delta / scale
    q75 = float(np.percentile(arr, 75))

    return np.asarray(
        [
            mean_val,
            std_val,
            min_val,
            max_val,
            slope_norm,
            delta_norm,
            q75,
            float(arr[-1]),
        ],
        dtype=np.float64,
    )


def extract_policy_signal_map(
    model_input_dir: Path,
    fold_ids: list[int],
    window: int,
) -> dict[tuple[int, str], np.ndarray]:
    output: dict[tuple[int, str], np.ndarray] = {}
    for fold_id in fold_ids:
        path = model_input_dir / f"fold_{fold_id}" / f"window_{window}" / "test.json"
        if not path.exists():
            continue
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(records, list):
            continue

        for record in records:
            if not isinstance(record, dict):
                continue
            doc_id = clean_text(record.get("doc_id", ""))
            if not doc_id:
                continue
            time_series = record.get("positive_time_series", None)
            if (
                not isinstance(time_series, list)
                or not time_series
                or not isinstance(time_series[0], list)
            ):
                continue
            try:
                vec = np.asarray(time_series[0], dtype=np.float64)
            except Exception:
                continue
            if vec.size < 2:
                continue
            output[(int(fold_id), doc_id)] = vec
    return output


def extract_location_slice(
    frame: pd.DataFrame,
    location: str,
    policy_year: int,
    history_window: int,
) -> pd.DataFrame:
    loc = clean_text(location)
    if not loc:
        return pd.DataFrame()

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
        return pd.DataFrame()

    if "year" not in filtered.columns:
        return pd.DataFrame()

    end_year = int(policy_year) - 1
    start_year = end_year - int(history_window) + 1
    filtered = filtered[
        (filtered["year"] >= start_year) & (filtered["year"] <= end_year)
    ]
    if filtered.empty:
        return pd.DataFrame()
    return filtered.sort_values("year").reset_index(drop=True)


def table_signal_descriptor(
    table_csv_path: Path,
    country: str,
    region: str,
    policy_year: int,
    history_window: int,
    max_table_features: int,
) -> np.ndarray | None:
    if not table_csv_path.exists():
        return None
    try:
        frame = pd.read_csv(table_csv_path)
    except Exception:
        return None

    location_candidates = [clean_text(country), clean_text(region), "World"]
    descriptors: list[np.ndarray] = []
    for location in location_candidates:
        if not location:
            continue
        sliced = extract_location_slice(
            frame=frame,
            location=location,
            policy_year=policy_year,
            history_window=history_window,
        )
        if sliced.empty:
            continue
        numeric_cols = sliced.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col.lower() != "year"]
        if not numeric_cols:
            continue
        for col in numeric_cols[: max(int(max_table_features), 1)]:
            values = pd.to_numeric(sliced[col], errors="coerce").dropna().to_numpy()
            if values.size < 2:
                continue
            descriptors.append(summarize_series(values))

    if not descriptors:
        return None
    return np.mean(np.vstack(descriptors), axis=0).astype(np.float64)


def pearson_score(policy_descriptor: np.ndarray, table_descriptor: np.ndarray) -> float:
    if policy_descriptor.size == 0 or table_descriptor.size == 0:
        return -1.0
    try:
        corr = np.corrcoef(policy_descriptor, table_descriptor)[0, 1]
        if np.isnan(corr):
            return -1.0
        return float(corr)
    except Exception:
        return -1.0


def linear_trend_score(
    policy_signal: np.ndarray, table_descriptor: np.ndarray
) -> float:
    if policy_signal.size < 2 or table_descriptor.size < 6:
        return -1e9
    policy_desc = summarize_series(policy_signal)
    policy_slope = float(policy_desc[4])
    policy_delta = float(policy_desc[5])
    table_slope = float(table_descriptor[4])
    table_delta = float(table_descriptor[5])
    return -abs(policy_slope - table_slope) - 0.5 * abs(policy_delta - table_delta)


def select_titles_for_policies_statistical(
    policy_df: pd.DataFrame,
    titles: list[str],
    title_to_csv: dict[str, Path],
    policy_signal_map: dict[tuple[int, str], np.ndarray],
    top_n: int,
    fold_col: str,
    doc_id_col: str,
    history_window: int,
    selector_method: str,
    max_table_features: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    max_pick = max(int(top_n), 1)

    descriptor_cache: dict[tuple[str, str, str, int, int, int], np.ndarray | None] = {}

    for _, row in policy_df.iterrows():
        fold_id = int(row[fold_col])
        doc_id = clean_text(row[doc_id_col])
        policy_signal = policy_signal_map.get((fold_id, doc_id), np.asarray([]))
        policy_descriptor = (
            summarize_series(policy_signal)
            if policy_signal.size >= 2
            else np.asarray([])
        )

        policy_year = extract_policy_year(row)
        country = clean_text(row.get("Geography", ""))
        region = clean_text(row.get("region", ""))
        metadata_text = policy_metadata_text(row)

        scored: list[tuple[str, float]] = []
        if policy_year is not None:
            for title in titles:
                csv_path = title_to_csv.get(title, None)
                if csv_path is None:
                    continue
                cache_key = (
                    title,
                    country,
                    region,
                    int(policy_year),
                    int(history_window),
                    int(max_table_features),
                )
                if cache_key not in descriptor_cache:
                    descriptor_cache[cache_key] = table_signal_descriptor(
                        table_csv_path=csv_path,
                        country=country,
                        region=region,
                        policy_year=policy_year,
                        history_window=history_window,
                        max_table_features=max_table_features,
                    )
                table_descriptor = descriptor_cache[cache_key]
                if table_descriptor is None:
                    continue

                if selector_method == "pearson":
                    score = pearson_score(policy_descriptor, table_descriptor)
                else:
                    score = linear_trend_score(policy_signal, table_descriptor)
                scored.append((title, float(score)))

        scored.sort(key=lambda item: item[1], reverse=True)
        selected_titles = [title for title, _ in scored[:max_pick]]
        selected_scores = [score for _, score in scored[:max_pick]]

        rows.append(
            {
                fold_col: fold_id,
                doc_id_col: doc_id,
                "policy_metadata": metadata_text,
                "selected_titles": json.dumps(selected_titles, ensure_ascii=False),
                "selected_scores": json.dumps(selected_scores),
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-1 semantic gating utility for SAX-Align. "
            "Embeds OWID table titles and selects top-N titles for each policy metadata row."
        )
    )
    parser.add_argument("--policy-input", type=Path, default=DEFAULT_POLICY_INPUT)
    parser.add_argument("--owid-dir", type=Path, default=DEFAULT_OWID_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--cache-file", type=str, default=DEFAULT_CACHE_FILE)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument(
        "--selector-method",
        type=str,
        default="semantic",
        choices=["semantic", "llm-prompt", "pearson", "linear-trend"],
        help=(
            "Table selection method. semantic=embedding cosine; "
            "llm-prompt=prompt-only LLM title selection; "
            "pearson=correlation baseline; linear-trend=trend-matching baseline."
        ),
    )
    parser.add_argument(
        "--model-input-dir",
        type=Path,
        default=DEFAULT_MODEL_INPUT_DIR,
        help="Required for pearson/linear-trend methods to load policy time-series vectors.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_SELECTOR_WINDOW,
        help=(
            "Fixed reference window index for statistical selector baselines "
            "(pearson/linear-trend). Default=10."
        ),
    )
    parser.add_argument(
        "--max-table-features",
        type=int,
        default=6,
        help="Max numeric columns per table/location used in statistical selector baselines.",
    )
    parser.add_argument("--openai-concurrency", type=int, default=4)
    parser.add_argument("--fold-column", type=str, default=DEFAULT_FOLD_COLUMN)
    parser.add_argument("--doc-id-column", type=str, default=DEFAULT_DOC_ID_COLUMN)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument(
        "--run-all-methods",
        action="store_true",
        help="Run all selector methods in one command and save one output CSV per method.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files (default is no-overwrite).",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be > 0")
    if args.top_n <= 0:
        raise ValueError("--top-n must be > 0")
    if args.openai_concurrency <= 0:
        raise ValueError("--openai-concurrency must be > 0")
    if args.max_table_features <= 0:
        raise ValueError("--max-table-features must be > 0")

    methods_to_run = (
        ["semantic", "pearson", "linear-trend"]
        if args.run_all_methods
        else [args.selector_method]
    )

    if any(method in {"semantic"} for method in methods_to_run):
        ensure_openai_api_key_loaded()
        configure_openai_concurrency(args.openai_concurrency)

    cache_path = args.output_dir / args.cache_file

    policy_df = load_policy_df(
        path=args.policy_input,
        fold_col=args.fold_column,
        doc_id_col=args.doc_id_column,
    )
    if args.fold is not None:
        policy_df = policy_df[policy_df[args.fold_column] == int(args.fold)].copy()
    if args.max_rows is not None and args.max_rows > 0:
        policy_df = policy_df.head(int(args.max_rows)).copy()
    policy_df = policy_df.reset_index(drop=True)

    titles = discover_owid_titles(args.owid_dir)
    if not titles:
        raise RuntimeError(f"No OWID table titles found under {args.owid_dir}")

    embeddings: OpenAIEmbeddings | None = None
    title_vectors: np.ndarray | None = None
    if "semantic" in methods_to_run:
        embeddings = OpenAIEmbeddings(model=args.embedding_model)
        cache_payload = load_title_cache(cache_path)
        if cache_payload is not None:
            cached_titles, cached_vectors = cache_payload
            if cached_titles == titles:
                title_vectors = cached_vectors
                print(
                    f"[SELECTOR] Loaded title embedding cache from {cache_path} with {len(titles)} titles",
                    flush=True,
                )
            else:
                title_vectors = embed_texts(
                    embeddings, titles, args.embedding_batch_size
                )
                title_vectors = normalize_rows(title_vectors)
                save_title_cache(cache_path, titles, title_vectors)
                print(
                    "[SELECTOR] Cache titles changed; recomputed title embeddings.",
                    flush=True,
                )
        else:
            title_vectors = embed_texts(embeddings, titles, args.embedding_batch_size)
            title_vectors = normalize_rows(title_vectors)
            save_title_cache(cache_path, titles, title_vectors)
            print(
                f"[SELECTOR] Built and saved title embedding cache: {cache_path}",
                flush=True,
            )

    title_to_csv: dict[str, Path] = {}
    policy_signal_map: dict[tuple[int, str], np.ndarray] = {}
    if any(method in {"pearson", "linear-trend"} for method in methods_to_run):
        if args.window <= 0:
            raise ValueError("--window must be > 0")
        for file_path in sorted(args.owid_dir.rglob("*.meta.json")):
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
            if title:
                title_to_csv[title] = Path(str(file_path).replace(".meta.json", ".csv"))

        fold_ids = sorted(policy_df[args.fold_column].unique().tolist())
        policy_signal_map = extract_policy_signal_map(
            model_input_dir=args.model_input_dir,
            fold_ids=[int(fid) for fid in fold_ids],
            window=int(args.window),
        )

    run_summaries: list[dict[str, object]] = []
    total_folds = policy_df[args.fold_column].nunique()
    print(
        "[SELECTOR] "
        f"methods={methods_to_run} rows={len(policy_df)} folds={int(total_folds)}",
        flush=True,
    )

    for method in methods_to_run:
        method_start = time.perf_counter()
        print(
            "[SELECTOR] "
            f"running method={method} top_n={int(args.top_n)} window={int(args.window)}",
            flush=True,
        )

        if method == "semantic":
            if embeddings is None or title_vectors is None:
                raise RuntimeError(
                    "Semantic selector requested but embeddings were not prepared."
                )
            selected_df = select_titles_for_policies(
                policy_df=policy_df,
                titles=titles,
                title_vectors=title_vectors,
                embeddings=embeddings,
                top_n=args.top_n,
                batch_size=args.embedding_batch_size,
                fold_col=args.fold_column,
                doc_id_col=args.doc_id_column,
            )
        elif method == "llm-prompt":
            selected_df = select_titles_for_policies_llm_prompt(
                policy_df=policy_df,
                titles=titles,
                fold_col=args.fold_column,
                doc_id_col=args.doc_id_column,
                top_n=args.top_n,
                llm_model=args.llm_model,
            )
        else:
            selected_df = select_titles_for_policies_statistical(
                policy_df=policy_df,
                titles=titles,
                title_to_csv=title_to_csv,
                policy_signal_map=policy_signal_map,
                top_n=args.top_n,
                fold_col=args.fold_column,
                doc_id_col=args.doc_id_column,
                history_window=int(args.window),
                selector_method=method,
                max_table_features=int(args.max_table_features),
            )

        output_path = resolve_output_path(
            output_dir=args.output_dir,
            output_file=args.output_file,
            selector_method=method,
            fold=args.fold,
            run_all_methods=args.run_all_methods,
            prevent_overwrite=not args.overwrite,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        selected_df.to_csv(output_path, index=False)

        metadata = {
            "task": "sax_align_table_selector",
            "policy_input": str(args.policy_input),
            "owid_dir": str(args.owid_dir),
            "embedding_model": args.embedding_model,
            "llm_model": args.llm_model,
            "embedding_batch_size": int(args.embedding_batch_size),
            "selector_method": method,
            "run_all_methods": bool(args.run_all_methods),
            "model_input_dir": str(args.model_input_dir),
            "window": int(args.window),
            "max_table_features": int(args.max_table_features),
            "top_n": int(args.top_n),
            "fold": args.fold,
            "max_rows": args.max_rows,
            "rows": int(len(selected_df)),
            "titles": int(len(titles)),
            "output_csv": str(output_path),
            "cache_file": str(cache_path),
            "overwrite": bool(args.overwrite),
        }

        metadata_path = resolve_metadata_path(
            output_dir=args.output_dir,
            selector_method=method,
            run_all_methods=args.run_all_methods,
            prevent_overwrite=not args.overwrite,
        )
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        elapsed = time.perf_counter() - method_start
        print(
            "[SELECTOR] "
            f"done method={method} rows={len(selected_df)} elapsed={elapsed:.1f}s",
            flush=True,
        )
        print(f"[SELECTOR] Saved ({method}): {output_path}")
        run_summaries.append(
            {
                "method": method,
                "output_csv": str(output_path),
                "metadata_json": str(metadata_path),
                "rows": int(len(selected_df)),
            }
        )

    if args.run_all_methods:
        combined_path = args.output_dir / "run_metadata_all_methods.json"
        if not args.overwrite:
            combined_path = ensure_unique_path(combined_path)
        combined_metadata = {
            "task": "sax_align_table_selector_all_methods",
            "policy_input": str(args.policy_input),
            "methods": methods_to_run,
            "fold": args.fold,
            "window": int(args.window),
            "runs": run_summaries,
        }
        with combined_path.open("w", encoding="utf-8") as handle:
            json.dump(combined_metadata, handle, indent=2)
        print(f"[SELECTOR] Saved run summary: {combined_path}")


if __name__ == "__main__":
    main()
