import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

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
DEFAULT_OUTPUT_DIR = Path("results/seekpolicy_experiments")
DEFAULT_K_VALUES = "1,5,10"
DEFAULT_ENCODER = "sentence-transformers/all-distilroberta-v1"
DEFAULT_QUERY_COLUMN = "RAG_v1_summary"
DEFAULT_HUMAN_SUMMARY_COLUMN = "Family Summary"
DEFAULT_DOC_ID_COLUMN = "Document ID"
DEFAULT_FOLD_COLUMN = "fold"
DEFAULT_TEXT_PATH_COLUMN = "text_file_path"
DEFAULT_CHROMA_DIR = Path("data/vectorstore/seekpolicy_chroma")
DEFAULT_CHROMA_COLLECTION = "seekpolicy_chunks"
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200


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


class SemanticRetriever:
    def __init__(self, corpus_texts: list[str], doc_ids: list[str], model_name: str):
        self.model = SentenceTransformer(model_name)
        self.corpus_texts = corpus_texts
        self.doc_ids = doc_ids
        self.corpus_embeddings = self.model.encode(
            corpus_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

    def retrieve(self, query: str, k: int) -> list[str]:
        if not self.corpus_texts:
            return []

        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        hits = util.semantic_search(
            query_embedding,
            self.corpus_embeddings,
            top_k=len(self.corpus_texts),
        )

        scores = np.zeros(len(self.corpus_texts), dtype=np.float64)
        for hit in hits[0]:
            scores[int(hit["corpus_id"])] = float(hit["score"])

        ranked = rank_from_scores(scores, self.doc_ids)
        return ranked[:k]


class BM25Retriever:
    def __init__(self, corpus_texts: list[str], doc_ids: list[str]):
        self.doc_ids = doc_ids
        self.corpus_texts = corpus_texts
        tokenized_corpus = [simple_tokenize(doc) for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int) -> list[str]:
        if not self.corpus_texts:
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
    fold_df: pd.DataFrame,
    query_col: str,
    doc_id_col: str,
    retriever: SemanticRetriever | BM25Retriever,
    k_values: list[int],
) -> dict[str, float]:
    records = {f"hit@{k}": [] for k in k_values}
    records.update({f"precision@{k}": [] for k in k_values})
    records.update({f"ndcg@{k}": [] for k in k_values})

    skipped = 0
    max_k = max(k_values)

    for _, row in fold_df.iterrows():
        query = clean_text(row.get(query_col, ""))
        target = str(row.get(doc_id_col, ""))
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

    metrics["queries_evaluated"] = float(max(len(fold_df) - skipped, 0))
    metrics["queries_total"] = float(len(fold_df))
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
        raise ValueError(f"Baseline method `{baseline_method}` has no fold results.")

    stat_rows: list[dict] = []
    for candidate_method in sorted(results_df["method"].unique().tolist()):
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


def resolve_text_path(raw_path: str, text_base_dir: Path) -> Optional[Path]:
    value = str(raw_path).strip()
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (text_base_dir / candidate).resolve()


def read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def build_or_load_chroma_store(
    df: pd.DataFrame,
    chroma_dir: Path,
    collection_name: str,
    text_path_col: str,
    doc_id_col: str,
    text_base_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    rebuild: bool,
) -> object:
    try:
        from langchain_chroma import Chroma
    except Exception:
        from langchain_community.vectorstores import Chroma

    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
        from langchain_openai import OpenAIEmbeddings
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Missing dependencies for Chroma chunk store. Install with `pip install langchain-openai langchain-community langchain-chroma chromadb`."
        ) from exc

    if text_path_col not in df.columns:
        raise ValueError(
            f"Column `{text_path_col}` is required to build Chroma chunks."
        )

    embeddings = OpenAIEmbeddings()

    if chroma_dir.exists() and not rebuild:
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(chroma_dir),
        )

    if chroma_dir.exists() and rebuild:
        shutil.rmtree(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    dedup_cols = [doc_id_col, text_path_col]
    optional_cols = ["Geography", "Sector"]
    for optional_col in optional_cols:
        if optional_col in df.columns:
            dedup_cols.append(optional_col)

    source_df = df[dedup_cols].drop_duplicates().reset_index(drop=True)
    documents: list[Document] = []
    for _, row in source_df.iterrows():
        raw_path = row.get(text_path_col, "")
        text_path = resolve_text_path(str(raw_path), text_base_dir)
        if text_path is None or not text_path.exists():
            continue

        try:
            page_content = read_text_file(text_path)
        except Exception:
            continue

        page_content = clean_text(page_content)
        if not page_content:
            continue

        metadata = {
            "document_id": str(row.get(doc_id_col, "")),
            "source": str(text_path),
            "geography": clean_text(row.get("Geography", "")),
            "sector": clean_text(row.get("Sector", "")),
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    if not documents:
        raise RuntimeError(
            "No documents available to build Chroma chunks. Check text paths and input columns."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunk_docs = splitter.split_documents(documents)

    for chunk in chunk_docs:
        doc_id = str(chunk.metadata.get("document_id", ""))
        start_index = chunk.metadata.get("start_index", 0)
        chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{start_index}"

    store = Chroma.from_documents(
        documents=chunk_docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(chroma_dir),
    )
    if hasattr(store, "persist"):
        try:
            store.persist()
        except Exception:
            pass
    return store


def generate_rag_summaries(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    model_name: str,
    rag_k: int,
    doc_id_col: str,
    text_path_col: str,
    text_base_dir: Path,
    chroma_dir: Path,
    chroma_collection: str,
    chunk_size: int,
    chunk_overlap: int,
    rebuild_chroma: bool,
    use_metadata_filter: bool,
) -> pd.DataFrame:
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not openai_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to `.env`.")

    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Missing dependencies for RAG generation. Install with `pip install langchain-openai langchain-community langchain-chroma chromadb`."
        ) from exc

    vector_store = build_or_load_chroma_store(
        df=df,
        chroma_dir=chroma_dir,
        collection_name=chroma_collection,
        text_path_col=text_path_col,
        doc_id_col=doc_id_col,
        text_base_dir=text_base_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        rebuild=rebuild_chroma,
    )

    llm = ChatOpenAI(model=model_name, temperature=0)

    out = df.copy()
    if target_col not in out.columns:
        out[target_col] = ""

    for idx in out.index:
        existing = clean_text(out.at[idx, target_col])
        if existing:
            continue

        source_text = clean_text(out.at[idx, source_col])
        if not source_text:
            continue

        filter_query: dict[str, str] | None = None
        if use_metadata_filter:
            geo = (
                clean_text(out.at[idx, "Geography"])
                if "Geography" in out.columns
                else ""
            )
            sector = (
                clean_text(out.at[idx, "Sector"]) if "Sector" in out.columns else ""
            )

            metadata_filter: dict[str, str] = {}
            if geo:
                metadata_filter["geography"] = geo
            if sector:
                metadata_filter["sector"] = sector
            filter_query = metadata_filter if metadata_filter else None

        try:
            docs = vector_store.similarity_search_with_score(
                source_text,
                k=rag_k,
                filter=filter_query,
            )
        except Exception:
            docs = []

        retrieved_chunks: list[str] = []
        for item in docs:
            doc = item[0]
            chunk = clean_text(getattr(doc, "page_content", ""))
            if chunk:
                retrieved_chunks.append(chunk)

        context = "\n\n".join(retrieved_chunks[:rag_k])
        prompt = (
            "You are a policy analyst. Write one concise paragraph summary that blends the policy text "
            "with the retrieved evidence. Avoid naming specific policy titles unless necessary.\n\n"
            f"Policy text:\n{source_text}\n\n"
            f"Retrieved context:\n{context}"
        )
        response = llm.invoke(prompt)
        out.at[idx, target_col] = clean_text(response.content)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SEEK-Policy RAG summary pipeline with optional persisted Chroma/OpenAI generation, "
            "fold-aware retrieval metrics, and paired significance tests."
        )
    )

    parser.add_argument("--policy-input", type=Path, default=DEFAULT_POLICY_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-values", type=str, default=DEFAULT_K_VALUES)
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER)
    parser.add_argument("--fold", type=int, default=None)

    parser.add_argument("--fold-column", type=str, default=DEFAULT_FOLD_COLUMN)
    parser.add_argument("--doc-id-column", type=str, default=DEFAULT_DOC_ID_COLUMN)
    parser.add_argument(
        "--human-summary-column", type=str, default=DEFAULT_HUMAN_SUMMARY_COLUMN
    )
    parser.add_argument("--query-column", type=str, default=DEFAULT_QUERY_COLUMN)

    parser.add_argument(
        "--generate-rag-summary",
        action="store_true",
        help="Generate missing RAG summaries in --query-column using persisted Chroma + OpenAI.",
    )
    parser.add_argument(
        "--source-column",
        type=str,
        default=DEFAULT_HUMAN_SUMMARY_COLUMN,
        help="Source text column used when --generate-rag-summary is enabled.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model used for RAG summary generation.",
    )
    parser.add_argument("--rag-k", type=int, default=5)
    parser.add_argument(
        "--text-path-column", type=str, default=DEFAULT_TEXT_PATH_COLUMN
    )
    parser.add_argument("--text-base-dir", type=Path, default=Path("."))
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument(
        "--chroma-collection", type=str, default=DEFAULT_CHROMA_COLLECTION
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--rebuild-chroma",
        action="store_true",
        help="Rebuild persisted Chroma index from source policy documents.",
    )
    parser.add_argument(
        "--use-generation-metadata-filter",
        action="store_true",
        help=(
            "Apply geography/sector metadata filter during RAG query generation. "
            "Disabled by default to reduce circular validation risk."
        ),
    )
    parser.add_argument(
        "--generated-output",
        type=Path,
        default=None,
        help="Optional CSV path to store generated RAG summaries.",
    )

    parser.add_argument(
        "--significance-baseline",
        type=str,
        default="rag_summary_semantic",
        help="Method name used as baseline in paired significance tests.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    k_values = parse_csv_ints(args.k_values)
    if any(k <= 0 for k in k_values):
        raise ValueError("All k-values must be positive")

    policy_df = load_policy_df(
        path=args.policy_input,
        fold_col=args.fold_column,
        doc_id_col=args.doc_id_column,
        summary_col=args.human_summary_column,
        query_col=args.query_column,
    )

    if args.generate_rag_summary:
        policy_df = generate_rag_summaries(
            df=policy_df,
            source_col=args.source_column,
            target_col=args.query_column,
            model_name=args.openai_model,
            rag_k=args.rag_k,
            doc_id_col=args.doc_id_column,
            text_path_col=args.text_path_column,
            text_base_dir=args.text_base_dir,
            chroma_dir=args.chroma_dir,
            chroma_collection=args.chroma_collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            rebuild_chroma=args.rebuild_chroma,
            use_metadata_filter=args.use_generation_metadata_filter,
        )
        generated_output = args.generated_output
        if generated_output is None:
            generated_output = args.output_dir / "generated_rag_summaries.csv"
        generated_output.parent.mkdir(parents=True, exist_ok=True)
        policy_df.to_csv(generated_output, index=False)

    fold_ids = sorted(policy_df[args.fold_column].unique().tolist())
    if args.fold is not None:
        if args.fold not in fold_ids:
            raise ValueError(
                f"Fold {args.fold} does not exist in {args.policy_input} for column `{args.fold_column}`"
            )
        fold_ids = [args.fold]

    all_results: list[dict] = []
    for fold_id in fold_ids:
        train_df = policy_df[policy_df[args.fold_column] != fold_id][
            [args.human_summary_column, args.doc_id_column]
        ].copy()
        test_df = policy_df[policy_df[args.fold_column] == fold_id][
            [args.query_column, args.human_summary_column, args.doc_id_column]
        ].copy()

        train_df = train_df.drop_duplicates().reset_index(drop=True)
        test_df = test_df.drop_duplicates().reset_index(drop=True)

        corpus_texts = train_df[args.human_summary_column].astype(str).tolist()
        corpus_doc_ids = train_df[args.doc_id_column].astype(str).tolist()

        rag_semantic = SemanticRetriever(
            corpus_texts=corpus_texts,
            doc_ids=corpus_doc_ids,
            model_name=args.encoder,
        )
        human_semantic = SemanticRetriever(
            corpus_texts=corpus_texts,
            doc_ids=corpus_doc_ids,
            model_name=args.encoder,
        )
        bm25 = BM25Retriever(corpus_texts=corpus_texts, doc_ids=corpus_doc_ids)

        method_specs: list[tuple[str, SemanticRetriever | BM25Retriever, str]] = [
            ("rag_summary_semantic", rag_semantic, args.query_column),
            ("human_summary_semantic", human_semantic, args.human_summary_column),
            ("climate_policy_radar_bm25", bm25, args.human_summary_column),
        ]

        for method_name, retriever, query_col in method_specs:
            result = evaluate_fold(
                fold_df=test_df,
                query_col=query_col,
                doc_id_col=args.doc_id_column,
                retriever=retriever,
                k_values=k_values,
            )
            result["fold"] = int(fold_id)
            result["method"] = method_name
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
        "task": "seekpolicy_rag_summary_experiments",
        "policy_input": str(args.policy_input),
        "fold_column": args.fold_column,
        "doc_id_column": args.doc_id_column,
        "human_summary_column": args.human_summary_column,
        "query_column": args.query_column,
        "generated_rag_summary": bool(args.generate_rag_summary),
        "generation_metadata_filter": bool(args.use_generation_metadata_filter),
        "text_path_column": args.text_path_column,
        "chroma_dir": str(args.chroma_dir),
        "chroma_collection": args.chroma_collection,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "rebuild_chroma": bool(args.rebuild_chroma),
        "openai_model": args.openai_model,
        "folds": fold_ids,
        "encoder": args.encoder,
        "k_values": k_values,
        "methods": [
            "rag_summary_semantic",
            "human_summary_semantic",
            "climate_policy_radar_bm25",
        ],
        "metrics": ["Hit@k", "Precision@k", "NDCG@k"],
        "significance_baseline": args.significance_baseline,
        "results_csv": str(results_path),
        "significance_csv": str(significance_path),
    }

    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved results: {results_path}")
    print(f"Saved significance tests: {significance_path}")


if __name__ == "__main__":
    main()
