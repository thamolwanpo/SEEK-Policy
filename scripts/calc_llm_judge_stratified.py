"""Stratified Random Sampling + LLM-as-a-Judge evaluation.

Selects samples from retrieval traces using stratified sampling by hit@1
(success / failure) with best-effort geographic diversity, then evaluates
both the SEEK-Policy generated query *and* the human Family Summary for the
same document using deepeval GEval (Temporal-Semantic Alignment) and
HallucinationMetric side-by-side.

Why stratified sampling?
  Simple random sampling risks "cherry-picking" easy cases.  Stratifying by
  retrieval success / failure and geography proves to reviewers that the
  evaluation covers the full diversity of the POLiMATCH dataset.

Sampling protocol (default: 50 samples total):
  * 25 samples where SEEK-Policy achieved hit@1 = 1  (success)
  * 25 samples where SEEK-Policy achieved hit@1 = 0  (failure)
  Within each stratum, samples are drawn using round-robin across continents
  (Europe, Asia, Americas, Africa, Oceania) for geographic diversity.

Judge rubric (per deepeval metric):
  1. Temporal-Semantic Alignment (GEval 1-5):
       Does the summary accurately reflect the quantitative trends
       (upward / downward / stable) of the raw time-series data?
  2. Hallucination (HallucinationMetric 0-1):
       Does the summary claim a policy trend that is not supported by the
       provided source context?

Both metrics are applied to:
  - SEEK-Policy generated query  (summary_type = "seek_policy")
  - Human Family Summary         (summary_type = "human")
for the *same* document, enabling a direct comparison.

Usage:
    # Step 1 – stratified sampling only (no API key required):
    python scripts/calc_llm_judge_stratified.py \\
        --retrieval-traces-dir results/seekpolicy_experiments/ \\
        --policy-input data/csv/group_kfold_assignments.csv \\
        --output-dir results/llm_judge

    # Step 2 – also run deepeval evaluation (requires OPENAI_API_KEY):
    python scripts/calc_llm_judge_stratified.py \\
        --retrieval-traces-dir results/seekpolicy_experiments/ \\
        --policy-input data/csv/group_kfold_assignments.csv \\
        --output-dir results/llm_judge \\
        --run-evaluation \\
        --judge-model gpt-4o

    # Optional: use raw time-series features as judge input context:
        --model-input-dir data/model_input/kfold
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

try:
    from deepeval.metrics import GEval, HallucinationMetric
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Country → Continent lookup (based on countries list in scripts/utils/data.py)
# ---------------------------------------------------------------------------
COUNTRY_TO_CONTINENT: dict[str, str] = {
    # Europe
    "Albania": "Europe",
    "Andorra": "Europe",
    "Austria": "Europe",
    "Belarus": "Europe",
    "Belgium": "Europe",
    "Bosnia and Herzegovina": "Europe",
    "Bulgaria": "Europe",
    "Croatia": "Europe",
    "Cyprus": "Europe",
    "Czechia": "Europe",
    "Denmark": "Europe",
    "Estonia": "Europe",
    "European Union": "Europe",
    "Finland": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Greece": "Europe",
    "Hungary": "Europe",
    "Iceland": "Europe",
    "Ireland": "Europe",
    "Italy": "Europe",
    "Kosovo": "Europe",
    "Latvia": "Europe",
    "Liechtenstein": "Europe",
    "Lithuania": "Europe",
    "Luxembourg": "Europe",
    "Malta": "Europe",
    "Moldova": "Europe",
    "Monaco": "Europe",
    "Montenegro": "Europe",
    "Netherlands": "Europe",
    "North Macedonia (Republic of North Macedonia)": "Europe",
    "Norway": "Europe",
    "Poland": "Europe",
    "Portugal": "Europe",
    "Romania": "Europe",
    "Russia": "Europe",
    "San Marino": "Europe",
    "Serbia": "Europe",
    "Slovakia": "Europe",
    "Slovenia": "Europe",
    "Spain": "Europe",
    "Sweden": "Europe",
    "Switzerland": "Europe",
    "Ukraine": "Europe",
    "United Kingdom": "Europe",
    # Asia
    "Afghanistan": "Asia",
    "Armenia": "Asia",
    "Azerbaijan": "Asia",
    "Bahrain": "Asia",
    "Bangladesh": "Asia",
    "Bhutan": "Asia",
    "Brunei Darussalam": "Asia",
    "Cambodia": "Asia",
    "China": "Asia",
    "Georgia": "Asia",
    "Hong Kong": "Asia",
    "India": "Asia",
    "Indonesia": "Asia",
    "Iran": "Asia",
    "Israel": "Asia",
    "Japan": "Asia",
    "Jordan": "Asia",
    "Kazakhstan": "Asia",
    "Korea, North": "Asia",
    "Kuwait": "Asia",
    "Kyrgyzstan": "Asia",
    "Lao People's Democratic Republic": "Asia",
    "Lebanon": "Asia",
    "Malaysia": "Asia",
    "Maldives": "Asia",
    "Mongolia": "Asia",
    "Myanmar": "Asia",
    "Nepal": "Asia",
    "Oman": "Asia",
    "Pakistan": "Asia",
    "Palestine": "Asia",
    "Philippines": "Asia",
    "Qatar": "Asia",
    "Saudi Arabia": "Asia",
    "Singapore": "Asia",
    "South Korea": "Asia",
    "Sri Lanka": "Asia",
    "Syria": "Asia",
    "Taiwan": "Asia",
    "Tajikistan": "Asia",
    "Thailand": "Asia",
    "Timor-Leste": "Asia",
    "Turkey": "Asia",
    "Turkmenistan": "Asia",
    "United Arab Emirates": "Asia",
    "Uzbekistan": "Asia",
    "Vietnam": "Asia",
    "Yemen": "Asia",
    # Americas
    "Antigua and Barbuda": "Americas",
    "Argentina": "Americas",
    "Bahamas, The": "Americas",
    "Barbados": "Americas",
    "Belize": "Americas",
    "Bolivia": "Americas",
    "Brazil": "Americas",
    "Canada": "Americas",
    "Chile": "Americas",
    "Colombia": "Americas",
    "Costa Rica": "Americas",
    "Cuba": "Americas",
    "Dominica": "Americas",
    "Dominican Republic": "Americas",
    "Ecuador": "Americas",
    "Grenada": "Americas",
    "Guyana": "Americas",
    "Haiti": "Americas",
    "Jamaica": "Americas",
    "Mexico": "Americas",
    "Peru": "Americas",
    "Saint Kitts and Nevis": "Americas",
    "Saint Lucia": "Americas",
    "Saint Vincent and the Grenadines": "Americas",
    "Suriname": "Americas",
    "Trinidad and Tobago": "Americas",
    "United States of America": "Americas",
    "Uruguay": "Americas",
    "Venezuela": "Americas",
    # Africa
    "Algeria": "Africa",
    "Angola": "Africa",
    "Benin": "Africa",
    "Botswana": "Africa",
    "Burkina Faso": "Africa",
    "Burundi": "Africa",
    "Cabo Verde": "Africa",
    "Cameroon": "Africa",
    "Central African Republic": "Africa",
    "Chad": "Africa",
    "Côte d'Ivoire": "Africa",
    "Djibouti": "Africa",
    "Egypt": "Africa",
    "Equatorial Guinea": "Africa",
    "Eritrea": "Africa",
    "Eswatini": "Africa",
    "Ethiopia": "Africa",
    "Gabon": "Africa",
    "Gambia": "Africa",
    "Ghana": "Africa",
    "Guinea": "Africa",
    "Guinea-Bissau": "Africa",
    "Kenya": "Africa",
    "Lesotho": "Africa",
    "Liberia": "Africa",
    "Libya": "Africa",
    "Madagascar": "Africa",
    "Malawi": "Africa",
    "Mali": "Africa",
    "Mauritius": "Africa",
    "Morocco": "Africa",
    "Mozambique": "Africa",
    "Namibia": "Africa",
    "Niger": "Africa",
    "Nigeria": "Africa",
    "Rwanda": "Africa",
    "Sao Tome and Principe": "Africa",
    "Seychelles": "Africa",
    "Sierra Leone": "Africa",
    "Somalia": "Africa",
    "South Africa": "Africa",
    "South Sudan": "Africa",
    "Sudan": "Africa",
    "Tanzania": "Africa",
    "Togo": "Africa",
    "Tunisia": "Africa",
    "Uganda": "Africa",
    "Zambia": "Africa",
    "Zimbabwe": "Africa",
    # Oceania
    "Australia": "Oceania",
    "Cook Islands": "Oceania",
    "Fiji": "Oceania",
    "Kiribati": "Oceania",
    "Marshall Islands": "Oceania",
    "Micronesia": "Oceania",
    "Nauru": "Oceania",
    "New Zealand": "Oceania",
    "Niue": "Oceania",
    "Palau": "Oceania",
    "Papua New Guinea": "Oceania",
    "Samoa": "Oceania",
    "Solomon Islands": "Oceania",
    "Tonga": "Oceania",
    "Tuvalu": "Oceania",
    "Vanuatu": "Oceania",
    # Other
    "International": "Other",
    "No Geography": "Other",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stratified sampling + LLM-as-a-Judge evaluation. "
            "Selects balanced success/failure samples with geographic diversity "
            "and evaluates SEEK-Policy generated queries vs. human summaries "
            "using deepeval GEval and HallucinationMetric."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--retrieval-traces-dir",
        type=Path,
        required=True,
        help="Directory containing retrieval_traces.csv files (per fold/window).",
    )
    parser.add_argument(
        "--policy-input",
        type=Path,
        required=True,
        help="CSV with human summaries (e.g., group_kfold_assignments.csv).",
    )
    parser.add_argument(
        "--query-column",
        type=str,
        default="RAG_v1_summary",
        help=(
            "Column name for the SEEK-Policy generated query. "
            "Used when loading from generated-summaries files; "
            "retrieval_traces.csv always uses the 'query_text' column."
        ),
    )
    parser.add_argument(
        "--human-summary-column",
        type=str,
        default="Family Summary",
        help="Column in policy-input CSV that holds the human-written summary.",
    )
    parser.add_argument(
        "--doc-id-column",
        type=str,
        default="Document ID",
        help="Column name for the document ID.",
    )
    parser.add_argument(
        "--n-success",
        type=int,
        default=25,
        help="Number of hit@1 = 1 (success) samples to include.",
    )
    parser.add_argument(
        "--n-failure",
        type=int,
        default=25,
        help="Number of hit@1 = 0 (failure) samples to include.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/llm_judge"),
        help="Output directory for the sampled CSV and evaluation results.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="LLM model used as judge (e.g., gpt-4o, claude-3-5-sonnet-20241022).",
    )
    parser.add_argument(
        "--model-input-dir",
        type=Path,
        default=None,
        help=(
            "Optional path to data/model_input/kfold. "
            "When provided, raw time-series features from test.json are formatted "
            "as statistical descriptions and used as judge input context instead "
            "of the retrieved chunk text."
        ),
    )
    parser.add_argument(
        "--run-evaluation",
        action="store_true",
        help=(
            "Run deepeval LLM-as-a-Judge scoring after sampling. "
            "Requires deepeval to be installed and OPENAI_API_KEY to be set."
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="If set, restrict sampling to traces from this window size only.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_window_from_path(path: str) -> str:
    m = re.search(r"window_(\d+)", path)
    return m.group(1) if m else "unknown"


def assign_continent(geography: str) -> str:
    """Map a country / geography string to a continent label."""
    if not isinstance(geography, str):
        return "Other"
    return COUNTRY_TO_CONTINENT.get(geography.strip(), "Other")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_retrieval_traces(retrieval_traces_dir: Path) -> pd.DataFrame:
    """Walk the directory tree and load all *retrieval_traces* CSV files.

    Expected columns (written by 4_seekpolicy.py --save-retrieval-traces):
        method, fold, window, query_index, query_text, target_doc_id,
        retrieved_rank, retrieved_doc_id, retrieved_score, is_relevant_doc,
        retrieved_chunk_id, retrieved_chunk_source, retrieved_chunk_text
    """
    frames: list[pd.DataFrame] = []
    for root, _, files in os.walk(retrieval_traces_dir):
        for fname in files:
            if not fname.endswith(".csv"):
                continue
            if "retrieval_traces" not in fname:
                continue
            fpath = Path(root) / fname
            try:
                df = pd.read_csv(fpath)
                required = {"target_doc_id", "retrieved_rank", "retrieved_doc_id"}
                if not required.issubset(df.columns):
                    continue
                if "window" not in df.columns:
                    df["window"] = extract_window_from_path(str(fpath))
                df["_trace_file"] = str(fpath)
                frames.append(df)
            except Exception as exc:
                print(f"Warning: skipping {fpath}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_model_input_map(
    model_input_dir: Path,
    fold: int,
    window: int,
) -> dict[str, dict]:
    """Load test.json for a given fold/window, keyed by doc_id."""
    path = model_input_dir / f"fold_{fold}" / f"window_{window}" / "test.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            records: list[dict] = json.load(fh)
        return {str(r.get("doc_id", "")): r for r in records if r.get("doc_id")}
    except Exception as exc:
        print(f"Warning: could not load model input {path}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Per-query hit@1 derivation
# ---------------------------------------------------------------------------


def compute_per_query_hits(traces: pd.DataFrame) -> pd.DataFrame:
    """Derive one row per unique query from ranked retrieval traces.

    A query is uniquely identified by (target_doc_id, fold, window, method).
    hit@1 = 1 when the rank-1 retrieved document matches the target.

    Returns a DataFrame with columns:
        target_doc_id, fold, window, method,
        query_text, hit_at_1,
        top_chunk_text, top_chunk_source
    """
    traces = traces.copy()
    traces["retrieved_rank"] = pd.to_numeric(
        traces["retrieved_rank"], errors="coerce"
    )

    group_keys = [
        c for c in ["fold", "window", "method", "target_doc_id"] if c in traces.columns
    ]

    rows: list[dict] = []
    for keys, grp in traces.groupby(group_keys, dropna=False):
        key_dict = dict(
            zip(group_keys, keys if isinstance(keys, tuple) else (keys,))
        )

        rank1 = grp[grp["retrieved_rank"] == 1]
        if rank1.empty:
            continue
        rank1_row = rank1.iloc[0]

        hit_at_1 = int(float(rank1_row.get("is_relevant_doc", 0)) == 1.0)

        query_text = ""
        if "query_text" in grp.columns:
            qt = grp["query_text"].dropna()
            query_text = str(qt.iloc[0]) if len(qt) > 0 else ""

        rows.append(
            {
                "target_doc_id": str(key_dict.get("target_doc_id", "")),
                "fold": key_dict.get("fold", ""),
                "window": key_dict.get("window", ""),
                "method": key_dict.get("method", ""),
                "query_text": query_text,
                "hit_at_1": hit_at_1,
                "top_chunk_text": str(rank1_row.get("retrieved_chunk_text", "")),
                "top_chunk_source": str(rank1_row.get("retrieved_chunk_source", "")),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stratified geographic sampling
# ---------------------------------------------------------------------------


def stratified_geographic_sample(
    pool: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample n rows with best-effort geographic diversity via round-robin.

    Rows are grouped by continent; the algorithm cycles through continents
    (alphabetically) and picks one row at a time until n rows are selected.
    Falls back to simple random sampling when continent info is unavailable.
    """
    if len(pool) <= n:
        return pool.copy()

    if "continent" not in pool.columns:
        idx = rng.choice(len(pool), size=n, replace=False)
        return pool.iloc[idx].reset_index(drop=True)

    # Shuffle within each continent group first
    shuffled: dict[str, pd.DataFrame] = {}
    for continent, grp in pool.groupby("continent"):
        perm = rng.permutation(len(grp))
        shuffled[continent] = grp.iloc[perm].reset_index(drop=True)

    selected: list[dict] = []
    continents = sorted(shuffled.keys())
    positions = {c: 0 for c in continents}

    while len(selected) < n:
        progress = False
        for c in continents:
            if len(selected) >= n:
                break
            pos = positions[c]
            if pos < len(shuffled[c]):
                selected.append(shuffled[c].iloc[pos].to_dict())
                positions[c] += 1
                progress = True
        if not progress:
            break  # all pools exhausted

    return pd.DataFrame(selected[:n])


# ---------------------------------------------------------------------------
# Judge input construction
# ---------------------------------------------------------------------------


def format_ts_context_from_record(record: dict) -> str:
    """Format a model-input record's time-series array as a textual description."""
    ts = record.get("positive_time_series", [])
    country = record.get("country", "Unknown")
    sector = record.get("sector", "Unknown")

    if not ts or not isinstance(ts, list):
        return f"Country: {country}. Sector: {sector}. (No time-series data available.)"

    ts_arr = np.array(ts, dtype=float)
    if ts_arr.ndim == 1:
        ts_arr = ts_arr.reshape(-1, 1)

    window_len, n_features = ts_arr.shape
    lines = [
        f"Country: {country}",
        f"Sector: {sector}",
        f"Time-series window: {window_len} timestep(s), {n_features} feature(s).",
    ]
    for feat_idx in range(n_features):
        col = ts_arr[:, feat_idx]
        first, last = float(col[0]), float(col[-1])
        delta = last - first
        ref = abs(first) + 1e-9
        if abs(delta) < 0.05 * ref:
            direction = "stable"
        elif delta > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        lines.append(
            f"  Feature {feat_idx + 1}: "
            f"start={first:.3f}, end={last:.3f}, "
            f"mean={float(col.mean()):.3f}, std={float(col.std()):.3f}, "
            f"trend={direction}"
        )
    return "\n".join(lines)


def build_judge_input(row: dict, model_input_map: dict[str, dict]) -> str:
    """Return the text used as 'input' for the LLM judge.

    Priority:
        1. Formatted TS statistics from model-input JSON (most informative).
        2. Top retrieved chunk text as proxy policy context.
        3. Minimal metadata description (fallback).
    """
    doc_id = str(row.get("target_doc_id", ""))
    if doc_id in model_input_map:
        return format_ts_context_from_record(model_input_map[doc_id])

    chunk_text = str(row.get("top_chunk_text", "")).strip()
    if chunk_text and chunk_text not in ("", "nan"):
        geography = str(row.get("Geography", "Unknown"))
        sector = str(row.get("Sector", "Unknown"))
        return (
            f"Country: {geography}. Sector: {sector}.\n"
            f"Retrieved policy context:\n{chunk_text}"
        )

    geography = str(row.get("Geography", "Unknown"))
    sector = str(row.get("Sector", "Unknown"))
    return f"Policy document for {geography} in the {sector} sector."


# ---------------------------------------------------------------------------
# deepeval evaluation
# ---------------------------------------------------------------------------


def run_deepeval(
    eval_df: pd.DataFrame,
    judge_model: str,
    model_input_dir: Path | None,
    output_dir: Path,
) -> None:
    """Run GEval + HallucinationMetric for every sample, comparing
    SEEK-Policy generated query vs. human summary for the same document.

    Outputs:
        results/llm_judge/llm_judge_results.csv   – per-sample scores
    """
    if not DEEPEVAL_AVAILABLE:
        print(
            "deepeval is not installed. "
            "Run `pip install deepeval` and re-run with --run-evaluation."
        )
        return

    alignment_metric = GEval(
        name="Temporal-Semantic Alignment",
        criteria=(
            "Does the summary accurately reflect the quantitative trends "
            "(upward / downward / stable) present in the raw time-series data "
            "described in the input? "
            "Penalise any directional claim that contradicts the data."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=judge_model,
    )
    hallucination_metric = HallucinationMetric(threshold=0.5, model=judge_model)

    # Cache model-input maps: (fold, window) → {doc_id: record}
    mi_cache: dict[tuple, dict[str, dict]] = {}

    results: list[dict] = []
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="LLM-as-a-Judge"):
        row_dict = row.to_dict()
        doc_id = str(row_dict.get("target_doc_id", ""))
        fold = row_dict.get("fold", "")
        window_val = row_dict.get("window", "")

        # Load model-input map for this fold/window (cached)
        mi_map: dict[str, dict] = {}
        if model_input_dir is not None:
            cache_key = (fold, window_val)
            if cache_key not in mi_cache:
                try:
                    mi_cache[cache_key] = load_model_input_map(
                        model_input_dir, int(fold), int(window_val)
                    )
                except (ValueError, TypeError):
                    mi_cache[cache_key] = {}
            mi_map = mi_cache[cache_key]

        judge_input = build_judge_input(row_dict, mi_map)
        chunk_text = str(row_dict.get("top_chunk_text", "")).strip()
        # Context for HallucinationMetric: the source material the summary
        # should be grounded in
        context_list = (
            [chunk_text] if chunk_text and chunk_text != "nan" else [judge_input]
        )

        base: dict = {
            "doc_id": doc_id,
            "fold": fold,
            "window": window_val,
            "stratum": row_dict.get("stratum", ""),
            "hit_at_1": row_dict.get("hit_at_1", ""),
            "Geography": row_dict.get("Geography", ""),
            "Sector": row_dict.get("Sector", ""),
            "continent": row_dict.get("continent", ""),
        }

        # Evaluate both summaries for this document
        for summary_type, summary_text in [
            ("seek_policy", str(row_dict.get("query_text", ""))),
            ("human", str(row_dict.get("human_summary", ""))),
        ]:
            if not summary_text.strip():
                continue

            row_result = {
                **base,
                "summary_type": summary_type,
                "summary_text": summary_text,
                "judge_input": judge_input,
            }

            # Temporal-Semantic Alignment
            try:
                tc_align = LLMTestCase(
                    input=judge_input, actual_output=summary_text
                )
                alignment_metric.measure(tc_align)
                row_result["alignment_score"] = alignment_metric.score
                row_result["alignment_reason"] = alignment_metric.reason
            except Exception as exc:
                row_result["alignment_score"] = None
                row_result["alignment_reason"] = f"ERROR: {exc}"

            # Hallucination
            try:
                tc_halluc = LLMTestCase(
                    input=judge_input,
                    actual_output=summary_text,
                    context=context_list,
                )
                hallucination_metric.measure(tc_halluc)
                row_result["hallucination_score"] = hallucination_metric.score
                row_result["hallucination_reason"] = getattr(
                    hallucination_metric, "reason", ""
                )
            except Exception as exc:
                row_result["hallucination_score"] = None
                row_result["hallucination_reason"] = f"ERROR: {exc}"

            results.append(row_result)

    if not results:
        print("No evaluation results produced.")
        return

    results_df = pd.DataFrame(results)
    out_path = output_dir / "llm_judge_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Saved LLM-as-a-Judge results to {out_path}")

    # Print mean-score summary comparing seek_policy vs. human
    score_cols = [c for c in ["alignment_score", "hallucination_score"] if c in results_df.columns]
    if score_cols:
        summary = (
            results_df.groupby("summary_type")[score_cols]
            .mean(numeric_only=True)
            .round(4)
        )
        print("\n=== LLM Judge – mean scores (seek_policy vs. human) ===")
        print(summary.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)

    # -------------------------------------------------------------------
    # 1. Load retrieval traces
    # -------------------------------------------------------------------
    print("Loading retrieval traces...")
    traces = load_retrieval_traces(args.retrieval_traces_dir)
    if traces.empty:
        print(
            f"No retrieval_traces.csv files found under {args.retrieval_traces_dir}.\n"
            "Run 4_seekpolicy.py with --save-retrieval-traces first."
        )
        return

    if args.window is not None and "window" in traces.columns:
        traces = traces[traces["window"].astype(str) == str(args.window)]
        if traces.empty:
            print(f"No traces found for window={args.window}.")
            return

    # -------------------------------------------------------------------
    # 2. Derive per-query hit@1
    # -------------------------------------------------------------------
    print("Computing per-query hit@1...")
    per_query = compute_per_query_hits(traces)
    if per_query.empty:
        print("Could not derive per-query metrics from the loaded traces.")
        return

    n_success_avail = int((per_query["hit_at_1"] == 1).sum())
    n_failure_avail = int((per_query["hit_at_1"] == 0).sum())
    print(
        f"  {len(per_query)} unique queries  |  "
        f"hit@1=1: {n_success_avail}  |  hit@1=0: {n_failure_avail}"
    )

    # -------------------------------------------------------------------
    # 3. Merge with policy data (human summary, Geography, Sector)
    # -------------------------------------------------------------------
    print("Merging with policy data...")
    policy_df = pd.read_csv(args.policy_input)
    required_cols = {args.doc_id_column, args.human_summary_column}
    if not required_cols.issubset(policy_df.columns):
        raise ValueError(
            f"Policy input is missing columns: "
            f"{required_cols - set(policy_df.columns)}"
        )
    policy_df[args.doc_id_column] = policy_df[args.doc_id_column].astype(str)

    keep_cols = [args.doc_id_column, args.human_summary_column] + [
        c for c in ["Geography", "Sector"] if c in policy_df.columns
    ]
    policy_map = (
        policy_df[keep_cols]
        .drop_duplicates(subset=[args.doc_id_column])
        .set_index(args.doc_id_column)
    )

    merged = per_query.join(policy_map, on="target_doc_id", how="left")
    merged = merged.rename(columns={args.human_summary_column: "human_summary"})

    # Assign continent for geographic stratification
    if "Geography" in merged.columns:
        merged["continent"] = merged["Geography"].apply(assign_continent)
    else:
        merged["continent"] = "Other"

    # Drop rows with no human summary (cannot evaluate without a reference)
    merged = merged[
        merged["human_summary"].notna() & (merged["human_summary"].str.strip() != "")
    ].reset_index(drop=True)
    print(f"  {len(merged)} queries with a human summary available.")

    # -------------------------------------------------------------------
    # 4. Stratified sampling: success vs. failure × geographic diversity
    # -------------------------------------------------------------------
    print("\nPerforming stratified sampling...")
    success_pool = merged[merged["hit_at_1"] == 1].reset_index(drop=True)
    failure_pool = merged[merged["hit_at_1"] == 0].reset_index(drop=True)

    n_success = min(args.n_success, len(success_pool))
    n_failure = min(args.n_failure, len(failure_pool))

    if n_success < args.n_success:
        print(
            f"  Warning: only {len(success_pool)} success samples available "
            f"(requested {args.n_success})."
        )
    if n_failure < args.n_failure:
        print(
            f"  Warning: only {len(failure_pool)} failure samples available "
            f"(requested {args.n_failure})."
        )

    success_sample = stratified_geographic_sample(success_pool, n_success, rng)
    success_sample = success_sample.copy()
    success_sample["stratum"] = "success"

    failure_sample = stratified_geographic_sample(failure_pool, n_failure, rng)
    failure_sample = failure_sample.copy()
    failure_sample["stratum"] = "failure"

    eval_df = pd.concat([success_sample, failure_sample], ignore_index=True)
    print(
        f"  Selected {len(success_sample)} success + {len(failure_sample)} failure "
        f"= {len(eval_df)} total samples."
    )

    if "continent" in eval_df.columns:
        continent_dist = (
            eval_df.groupby(["stratum", "continent"])
            .size()
            .unstack(fill_value=0)
        )
        print("\n  Geographic distribution across strata:")
        print(continent_dist.to_string())

    # -------------------------------------------------------------------
    # 5. Save validation set
    # -------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_path = args.output_dir / "validation_50_samples.csv"
    eval_df.to_csv(val_path, index=False)
    print(f"\nSaved validation set → {val_path}")

    # -------------------------------------------------------------------
    # 6. (Optional) LLM-as-a-Judge evaluation via deepeval
    # -------------------------------------------------------------------
    if args.run_evaluation:
        print("\nRunning LLM-as-a-Judge evaluation...")
        run_deepeval(
            eval_df=eval_df,
            judge_model=args.judge_model,
            model_input_dir=args.model_input_dir,
            output_dir=args.output_dir,
        )
    else:
        print(
            "\nTo run deepeval scoring, re-run with --run-evaluation "
            "(requires OPENAI_API_KEY and `pip install deepeval`)."
        )


if __name__ == "__main__":
    main()
