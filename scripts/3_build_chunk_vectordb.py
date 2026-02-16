import argparse
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def resolve_text_path(raw_path: str, text_base_dir: Path) -> Optional[Path]:
    value = str(raw_path).strip().replace("\\", "/")
    if not value:
        return None

    candidate = Path(value)
    checks: list[Path] = []
    if candidate.is_absolute():
        checks.append(candidate)
    else:
        checks.append((text_base_dir / candidate).resolve())
        checks.append((Path.cwd() / candidate).resolve())
        if value.startswith("../"):
            checks.append((Path("data") / value.removeprefix("../")).resolve())

    for path in checks:
        if path.exists() and path.is_file():
            return path
    return None


def load_policy_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"fold", "Document ID", "text_file_path", "Family Summary"}
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    out = df.copy()
    out["fold"] = out["fold"].astype(int)
    out["Document ID"] = out["Document ID"].astype(str)
    out["text_file_path"] = out["text_file_path"].astype(str)
    out["Family Summary"] = out["Family Summary"].apply(clean_text)
    out = out[out["Document ID"] != ""]
    return out.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a persisted chunk vector DB from policy text files using OpenAI embeddings. "
            "Chunks include fold and document metadata for fold-aware retrieval/evaluation."
        )
    )
    parser.add_argument(
        "--policy-input",
        type=Path,
        default=Path("data/csv/group_kfold_assignments.csv"),
    )
    parser.add_argument(
        "--vectordb-dir",
        type=Path,
        default=Path("data/vectorstore/policy_chunks_chroma"),
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="policy_chunks_openai",
    )
    parser.add_argument(
        "--text-path-base-dir",
        type=Path,
        default=Path("data/csv"),
    )
    parser.add_argument("--chunk-size", type=int, default=1500)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model used to build vector store.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=128,
        help="Number of chunks embedded per API batch for progress logging.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and rebuild persisted vector DB directory.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Optional cap for number of unique policy documents to ingest (0 = all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[INFO] Starting chunk vector DB build")

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if args.chunk_overlap < 0:
        raise ValueError("--chunk-overlap must be >= 0")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("--chunk-overlap must be smaller than --chunk-size")
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be > 0")
    print("[INFO] Argument validation complete")

    load_dotenv()
    print("[INFO] Environment variables loaded")
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to .env before building vector DB.")
    print("[INFO] OPENAI_API_KEY detected")

    try:
        from langchain_chroma import Chroma
    except Exception:
        from langchain_community.vectorstores import Chroma

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    print("[INFO] LangChain dependencies imported")

    embedding_progress_state = {
        "expected_total_chunks": 0,
        "embedded_chunks_done": 0,
    }

    class ProgressOpenAIEmbeddings(OpenAIEmbeddings):

        def embed_documents(self, texts: list[str], chunk_size: Optional[int] = None) -> list[list[float]]:
            total = len(texts)
            if total == 0:
                return []

            configured_batch_size = int(chunk_size or getattr(self, "chunk_size", 0) or args.embedding_batch_size)
            batch_size = configured_batch_size if configured_batch_size > 0 else args.embedding_batch_size

            vectors: list[list[float]] = []
            if (
                embedding_progress_state["embedded_chunks_done"] == 0
                and embedding_progress_state["expected_total_chunks"] > 0
            ):
                print(f"[INFO] Chunks left: {embedding_progress_state['expected_total_chunks']}")

            for start in range(0, total, batch_size):
                batch = texts[start : start + batch_size]
                vectors.extend(super().embed_documents(batch))
                embedding_progress_state["embedded_chunks_done"] += len(batch)
                left = max(
                    embedding_progress_state["expected_total_chunks"]
                    - embedding_progress_state["embedded_chunks_done"],
                    0,
                )
                print(f"[INFO] Chunks left: {left}")

            return vectors

    print(f"[INFO] Loading policy input from: {args.policy_input}")
    policy_df = load_policy_df(args.policy_input)
    print(f"[INFO] Loaded {len(policy_df)} rows from policy input")

    source_df = policy_df[["fold", "Document ID", "text_file_path", "Family Summary"]]
    source_df = source_df.drop_duplicates(subset=["Document ID"]).reset_index(drop=True)
    print(f"[INFO] Unique documents to process: {len(source_df)}")

    if args.max_docs > 0:
        source_df = source_df.head(args.max_docs).reset_index(drop=True)
        print(f"[INFO] Applied --max-docs={args.max_docs}; processing {len(source_df)} documents")

    documents: list[Document] = []
    skipped_files = 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        add_start_index=True,
    )
    print(
        f"[INFO] Text splitter initialized (chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap})"
    )

    total_docs = len(source_df)
    print(f"[INFO] Starting document chunking loop for {total_docs} documents")

    for idx, (_, row) in enumerate(source_df.iterrows(), start=1):
        if idx == 1 or idx % 100 == 0 or idx == total_docs:
            print(f"[INFO] Processing document {idx}/{total_docs}")

        fold = int(row["fold"])
        doc_id = str(row["Document ID"])
        text_path = resolve_text_path(str(row.get("text_file_path", "")), args.text_path_base_dir)
        raw_text = ""
        source_value = ""

        if text_path is not None:
            try:
                raw_text = text_path.read_text(encoding="utf-8")
                source_value = str(text_path)
            except Exception:
                raw_text = ""

        raw_text = clean_text(raw_text)
        if not raw_text:
            skipped_files += 1
            continue

        base_doc = Document(
            page_content=raw_text,
            metadata={
                "fold": fold,
                "document_id": doc_id,
                "source": source_value,
            },
        )

        chunk_docs = splitter.split_documents([base_doc])
        for chunk_doc in chunk_docs:
            start_index = int(chunk_doc.metadata.get("start_index", 0))
            chunk_doc.metadata["chunk_id"] = f"{doc_id}_chunk_{start_index}"
            chunk_doc.metadata["fold"] = fold
            chunk_doc.metadata["document_id"] = doc_id
            chunk_doc.metadata["source"] = source_value
            documents.append(chunk_doc)

    if not documents:
        raise RuntimeError("No chunk documents were built. Check input file paths and content.")
    embedding_progress_state["expected_total_chunks"] = len(documents)
    print(f"[INFO] Chunking complete: built {len(documents)} chunks, skipped {skipped_files} documents")

    if args.vectordb_dir.exists() and args.rebuild:
        print(f"[INFO] --rebuild enabled; removing existing vector DB at {args.vectordb_dir}")
        shutil.rmtree(args.vectordb_dir)
    args.vectordb_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Vector DB directory ready: {args.vectordb_dir}")

    print(f"[INFO] Initializing OpenAI embeddings model: {args.embedding_model}")
    embeddings = ProgressOpenAIEmbeddings(
        model=args.embedding_model,
        chunk_size=args.embedding_batch_size,
    )

    print(f"[INFO] Writing chunks to Chroma collection: {args.collection}")
    ingest_started_at = time.time()
    stop_progress = threading.Event()

    def log_ingest_progress() -> None:
        while not stop_progress.wait(15):
            elapsed = int(time.time() - ingest_started_at)
            print(
                f"[INFO] Chroma ingest in progress... elapsed={elapsed}s "
                f"(chunks queued={len(documents)})"
            )

    progress_thread = threading.Thread(target=log_ingest_progress, daemon=True)
    progress_thread.start()

    try:
        store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=args.collection,
            persist_directory=str(args.vectordb_dir),
        )
    finally:
        stop_progress.set()
        progress_thread.join(timeout=1)

    ingest_elapsed = int(time.time() - ingest_started_at)
    print(f"[INFO] Chroma vector store creation complete (elapsed={ingest_elapsed}s)")

    if hasattr(store, "persist"):
        try:
            store.persist()
            print("[INFO] Vector store persisted to disk")
        except Exception:
            pass

    manifest = {
        "policy_input": str(args.policy_input),
        "vectordb_dir": str(args.vectordb_dir),
        "collection": args.collection,
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "documents_ingested": int(source_df.shape[0]),
        "chunks_ingested": int(len(documents)),
        "skipped_documents": int(skipped_files),
        "folds": sorted(source_df["fold"].astype(int).unique().tolist()),
    }

    manifest_path = args.vectordb_dir / "manifest.json"
    print(f"[INFO] Writing manifest to: {manifest_path}")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Built vector DB at: {args.vectordb_dir}")
    print(f"Collection: {args.collection}")
    print(f"Chunks: {len(documents)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
