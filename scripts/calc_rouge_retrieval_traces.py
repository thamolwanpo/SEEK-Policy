import argparse
import os
import pandas as pd
from pathlib import Path
from rouge_score import rouge_scorer
from tqdm import tqdm


# Set up argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate ROUGE scores for generated queries vs. human summaries."
    )
    parser.add_argument(
        "--retrieval-traces-dir",
        type=Path,
        required=True,
        help="Directory containing retrieval traces (per fold/window)",
    )
    parser.add_argument(
        "--policy-input",
        type=Path,
        required=True,
        help="CSV file with human summaries (e.g., group_kfold_assignments.csv)",
    )
    parser.add_argument(
        "--query-column",
        type=str,
        default="RAG_v1_summary",
        help="Column name for generated query in retrieval traces",
    )
    parser.add_argument(
        "--human-summary-column",
        type=str,
        default="Family Summary",
        help="Column name for human summary in policy input",
    )
    parser.add_argument(
        "--doc-id-column",
        type=str,
        default="Document ID",
        help="Column name for document ID",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="results/rouge_scores.csv",
        help="Output CSV file for ROUGE scores",
    )
    return parser.parse_args()


# Load all retrieval trace files (per fold/window)
import re


def extract_window_from_path(path: str) -> str:
    # Looks for window_<number> in the path
    m = re.search(r"window_(\d+)", path)
    return m.group(1) if m else "unknown"


def load_retrieval_traces(
    retrieval_traces_dir: Path, query_column: str, doc_id_column: str
):
    records = []
    for root, _, files in os.walk(retrieval_traces_dir):
        for fname in files:
            # Only process files that look like retrieval traces
            if not (
                "retrieval_traces" in fname
                or fname.endswith(".csv")
                or fname.endswith(".jsonl")
            ):
                continue
            if fname.endswith(".json") and not "retrieval_traces" in fname:
                continue  # skip metadata/config jsons
            fpath = Path(root) / fname
            try:
                if fname.endswith(".csv"):
                    df = pd.read_csv(fpath)
                elif fname.endswith(".jsonl"):
                    df = pd.read_json(fpath, lines=True)
                else:
                    continue
                if query_column in df.columns and doc_id_column in df.columns:
                    window = extract_window_from_path(str(fpath))
                    for _, row in df.iterrows():
                        records.append(
                            {
                                "doc_id": str(row[doc_id_column]),
                                "query_text": str(row[query_column]),
                                "trace_file": str(fpath),
                                "window": window,
                            }
                        )
            except Exception as e:
                print(f"Warning: Failed to load {fpath}: {e}")
    return pd.DataFrame(records)


# Main scoring logic


def main():
    args = parse_args()
    traces_df = load_retrieval_traces(
        args.retrieval_traces_dir, args.query_column, args.doc_id_column
    )
    if traces_df.empty:
        print("No retrieval traces found.")
        return
    policy_df = pd.read_csv(args.policy_input)
    if (
        args.doc_id_column not in policy_df.columns
        or args.human_summary_column not in policy_df.columns
    ):
        raise ValueError(
            f"Missing columns in policy input: {args.doc_id_column}, {args.human_summary_column}"
        )
    policy_map = policy_df.set_index(args.doc_id_column)[
        args.human_summary_column
    ].to_dict()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    if "window" not in traces_df.columns:
        traces_df["window"] = "unknown"

    for window, window_df in traces_df.groupby("window"):
        results = []
        for _, row in tqdm(
            window_df.iterrows(), total=len(window_df), desc=f"Window {window}"
        ):
            doc_id = row["doc_id"]
            generated = row["query_text"]
            reference = policy_map.get(doc_id, "")
            if not reference:
                continue
            scores = scorer.score(reference, generated)
            results.append(
                {
                    "doc_id": doc_id,
                    "trace_file": row["trace_file"],
                    "rouge1": scores["rouge1"].fmeasure,
                    "rouge2": scores["rouge2"].fmeasure,
                    "rougeL": scores["rougeL"].fmeasure,
                    "query_text": generated,
                    "human_summary": reference,
                    "window": window,
                }
            )
        out_df = pd.DataFrame(results)
        out_path = (
            args.output.parent / f"rouge_scores_window_{window}.csv"
            if args.output.suffix == ".csv"
            else args.output
        )
        out_df.to_csv(out_path, index=False)
        print(f"Saved ROUGE scores for window {window} to {out_path}")


if __name__ == "__main__":
    main()
