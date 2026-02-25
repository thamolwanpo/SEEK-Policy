import argparse
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from bert_score import score
import re


# Set up argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate BERTScore for generated queries vs. human summaries."
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
        default="results/bertscore_scores.csv",
        help="Output CSV file for BERTScore scores",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language for BERTScore (default: en)",
    )
    return parser.parse_args()


def extract_window_from_path(path: str) -> str:
    m = re.search(r"window_(\d+)", path)
    return m.group(1) if m else "unknown"


def load_retrieval_traces(
    retrieval_traces_dir: Path, query_column: str, doc_id_column: str
):
    records = []
    for root, _, files in os.walk(retrieval_traces_dir):
        for fname in files:
            if not (
                "retrieval_traces" in fname
                or fname.endswith(".csv")
                or fname.endswith(".jsonl")
            ):
                continue
            if fname.endswith(".json") and not "retrieval_traces" in fname:
                continue
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

    if "window" not in traces_df.columns:
        traces_df["window"] = "unknown"

    for window, window_df in traces_df.groupby("window"):
        results = []
        candidates = []
        references = []
        doc_ids = []
        trace_files = []
        for _, row in tqdm(
            window_df.iterrows(), total=len(window_df), desc=f"Window {window}"
        ):
            doc_id = row["doc_id"]
            generated = row["query_text"]
            reference = policy_map.get(doc_id, "")
            if not reference:
                continue
            candidates.append(generated)
            references.append(reference)
            doc_ids.append(doc_id)
            trace_files.append(row["trace_file"])
        if candidates and references:
            P, R, F1 = score(candidates, references, lang=args.lang)
            for i in range(len(candidates)):
                results.append(
                    {
                        "doc_id": doc_ids[i],
                        "trace_file": trace_files[i],
                        "bertscore_precision": float(P[i]),
                        "bertscore_recall": float(R[i]),
                        "bertscore_f1": float(F1[i]),
                        "query_text": candidates[i],
                        "human_summary": references[i],
                        "window": window,
                    }
                )
        out_df = pd.DataFrame(results)
        out_path = (
            args.output.parent / f"bertscore_scores_window_{window}.csv"
            if args.output.suffix == ".csv"
            else args.output
        )
        out_df.to_csv(out_path, index=False)
        print(f"Saved BERTScore scores for window {window} to {out_path}")


if __name__ == "__main__":
    main()
