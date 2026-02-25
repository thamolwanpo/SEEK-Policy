import argparse
import json
import os
import random
import glob
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as f
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModel, AutoTokenizer


def safe_import_stats_tests() -> tuple[Optional[object], Optional[object]]:
    try:
        from scipy.stats import ttest_rel, wilcoxon  # type: ignore

        return ttest_rel, wilcoxon
    except Exception:
        return None, None


DEFAULT_POLICY_INPUT = Path("data/csv/group_kfold_assignments.csv")
DEFAULT_MODEL_INPUT_DIR = Path("data/model_input/kfold")
DEFAULT_OUTPUT_DIR = Path("results/retrieval_experiments")
DEFAULT_BACKBONES = "climatebert/distilroberta-base-climate-f,sentence-transformers/all-distilroberta-v1"
DEFAULT_LOSSES = "triplet,contrastive"
DEFAULT_WINDOWS = "1,2,5,10"
DEFAULT_K_VALUES = "1,5,10"
DEFAULT_CHUNK_VECTORDB_DIR = Path("data/vectorstore/policy_chunks_chroma")
DEFAULT_CHUNK_VECTORDB_COLLECTION = "policy_chunks_openai"
DEFAULT_TUNE_LR = "1e-4,3e-4"
DEFAULT_TUNE_WEIGHT_DECAY = "1e-4,1e-5"
DEFAULT_TUNE_TIME_SERIES_HIDDEN_SIZE = "256,512"
DEFAULT_TUNE_SECTOR_EMBEDDING_DIM = "16,32"
DEFAULT_TUNE_EMBEDDING_DIM = "256"
DEFAULT_TUNE_DROPOUT = "0.1,0.2"
DEFAULT_TUNE_MARGIN = "0.5,1.0"
DEFAULT_TUNE_TEMPERATURE = "0.05,0.07"
DEFAULT_SHARED_TUNED_HPARAMS_DIRNAME = "shared_tuned_hparams"
DEFAULT_TUNING_OUTPUT_DIRNAME = "tuning_runs"


@dataclass(frozen=True)
class ExperimentKey:
    fold: int
    window: int
    backbone: str
    loss: str


@dataclass(frozen=True)
class ModelHyperParams:
    time_series_hidden_size: int
    sector_embedding_dim: int
    embedding_dim: int
    dropout: float
    lr: float
    weight_decay: float
    margin: float
    temperature: float


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


def parse_csv_strs(raw: str) -> list[str]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one string value")
    return values


def parse_csv_floats(raw: str) -> list[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    values = sorted(set(values))
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate fold-aware retrieval dual-encoder experiments over "
            "multiple windows, backbones, and loss functions."
        )
    )

    parser.add_argument("--policy-input", type=Path, default=DEFAULT_POLICY_INPUT)
    parser.add_argument("--model-input-dir", type=Path, default=DEFAULT_MODEL_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--windows", type=str, default=DEFAULT_WINDOWS)
    parser.add_argument("--backbones", type=str, default=DEFAULT_BACKBONES)
    parser.add_argument("--losses", type=str, default=DEFAULT_LOSSES)
    parser.add_argument("--k-values", type=str, default=DEFAULT_K_VALUES)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--time-series-hidden-size", type=int, default=256)
    parser.add_argument("--sector-embedding-dim", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)

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
        "--tune-hyperparams",
        action="store_true",
        help="Run grid-search hyperparameter tuning on validation split before final training.",
    )
    parser.add_argument(
        "--tune-only",
        action="store_true",
        help="Run tuning only, save best hyperparameters, and skip final training/evaluation.",
    )
    parser.add_argument(
        "--use-tuned-hparams",
        action="store_true",
        help=(
            "Load best hyperparameters from run_dir/tuning_results.json and use them for final training. "
            "Useful after a separate --tune-only run."
        ),
    )
    parser.add_argument(
        "--shared-tuned-hparams",
        action="store_true",
        help=(
            "Enable shared tuned hyperparameters by backbone/loss across folds/windows. "
            "When tuning, saves best params to output-dir/shared_tuned_hparams/<backbone>/<loss>.json. "
            "When using --use-tuned-hparams, loads from the same shared location."
        ),
    )
    parser.add_argument(
        "--shared-tuned-hparams-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for shared tuned hyperparameter files. "
            "Defaults to <output-dir>/shared_tuned_hparams."
        ),
    )
    parser.add_argument("--tune-max-epochs", type=int, default=5)
    parser.add_argument("--tune-patience", type=int, default=2)
    parser.add_argument("--tune-max-trials", type=int, default=24)
    parser.add_argument("--tune-lr", type=str, default=DEFAULT_TUNE_LR)
    parser.add_argument(
        "--tune-weight-decay", type=str, default=DEFAULT_TUNE_WEIGHT_DECAY
    )
    parser.add_argument(
        "--tune-time-series-hidden-size",
        type=str,
        default=DEFAULT_TUNE_TIME_SERIES_HIDDEN_SIZE,
    )
    parser.add_argument(
        "--tune-sector-embedding-dim",
        type=str,
        default=DEFAULT_TUNE_SECTOR_EMBEDDING_DIM,
    )
    parser.add_argument(
        "--tune-embedding-dim", type=str, default=DEFAULT_TUNE_EMBEDDING_DIM
    )
    parser.add_argument("--tune-dropout", type=str, default=DEFAULT_TUNE_DROPOUT)
    parser.add_argument("--tune-margin", type=str, default=DEFAULT_TUNE_MARGIN)
    parser.add_argument(
        "--tune-temperature", type=str, default=DEFAULT_TUNE_TEMPERATURE
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Enable deterministic training behavior for reproducibility.",
    )
    parser.add_argument(
        "--non-deterministic",
        action="store_false",
        dest="deterministic",
        help="Allow non-deterministic kernels for speed.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate using existing checkpoints.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train and save checkpoints only; skip retrieval evaluation on test data.",
    )
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint path for --eval-only. "
            "If omitted, the script auto-discovers a final best checkpoint under output-dir."
        ),
    )
    parser.add_argument(
        "--save-retrieval-traces",
        action="store_true",
        help=(
            "Save per-query retrieved chunk traces to retrieval_traces.csv under each run directory."
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

    parser.add_argument(
        "--baseline-backbone",
        type=str,
        default="climatebert/distilroberta-base-climate-f",
    )
    parser.add_argument("--baseline-loss", type=str, default="triplet")
    parser.add_argument("--baseline-window", type=int, default=1)

    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_reproducibility(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    pl.seed_everything(seed, workers=True)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def detect_input_dims(train_jsonl: Path) -> tuple[int, int]:
    rows = load_jsonl(train_jsonl)
    if not rows:
        raise ValueError(f"No rows found in {train_jsonl}")

    sample = rows[0]
    ts = np.asarray(sample["positive_time_series"], dtype=np.float32)
    sector = np.asarray(sample["positive_sector"], dtype=np.float32)
    ts_input_dim = int(ts.reshape(-1).shape[0])
    sector_dim = int(sector.shape[0])

    if ts_input_dim <= 0 or sector_dim <= 0:
        raise ValueError(f"Invalid feature dimensions in {train_jsonl}")

    return ts_input_dim, sector_dim


def pick_text_embedding(output: object) -> torch.Tensor:
    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output

    last_hidden_state = getattr(output, "last_hidden_state", None)
    if last_hidden_state is None:
        raise ValueError(
            "Backbone output does not provide pooler_output or last_hidden_state"
        )
    return last_hidden_state[:, 0]


class TripletDataset(Dataset):
    def __init__(
        self, rows: list[dict], tokenizer: AutoTokenizer, max_length: int = 512
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        tokenized = self.tokenizer(
            row["anchor_summary"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        anchor = {k: v.squeeze(0) for k, v in tokenized.items()}

        return {
            "anchor_summary": anchor,
            "positive_time_series": torch.tensor(
                row["positive_time_series"], dtype=torch.float32
            ),
            "positive_sector": torch.tensor(
                row["positive_sector"], dtype=torch.float32
            ),
            "negative_time_series": torch.tensor(
                row["negative_time_series"], dtype=torch.float32
            ),
            "negative_sector": torch.tensor(
                row["negative_sector"], dtype=torch.float32
            ),
        }


class PositivePairDataset(Dataset):
    def __init__(
        self, rows: list[dict], tokenizer: AutoTokenizer, max_length: int = 512
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        tokenized = self.tokenizer(
            row["anchor_summary"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        anchor = {k: v.squeeze(0) for k, v in tokenized.items()}

        return {
            "anchor_summary": anchor,
            "positive_time_series": torch.tensor(
                row["positive_time_series"], dtype=torch.float32
            ),
            "positive_sector": torch.tensor(
                row["positive_sector"], dtype=torch.float32
            ),
        }


class TripletCollator:
    def __call__(self, batch: list[dict]) -> dict:
        return {
            "anchor_summary": {
                "input_ids": torch.stack(
                    [item["anchor_summary"]["input_ids"] for item in batch]
                ),
                "attention_mask": torch.stack(
                    [item["anchor_summary"]["attention_mask"] for item in batch]
                ),
            },
            "positive_time_series": torch.stack(
                [item["positive_time_series"] for item in batch]
            ),
            "positive_sector": torch.stack([item["positive_sector"] for item in batch]),
            "negative_time_series": torch.stack(
                [item["negative_time_series"] for item in batch]
            ),
            "negative_sector": torch.stack([item["negative_sector"] for item in batch]),
        }


class PositivePairCollator:
    def __call__(self, batch: list[dict]) -> dict:
        return {
            "anchor_summary": {
                "input_ids": torch.stack(
                    [item["anchor_summary"]["input_ids"] for item in batch]
                ),
                "attention_mask": torch.stack(
                    [item["anchor_summary"]["attention_mask"] for item in batch]
                ),
            },
            "positive_time_series": torch.stack(
                [item["positive_time_series"] for item in batch]
            ),
            "positive_sector": torch.stack([item["positive_sector"] for item in batch]),
        }


class RetrievalDualEncoder(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str,
        ts_input_dim: int,
        sector_dim: int,
        loss_name: str,
        time_series_hidden_size: int,
        sector_embedding_dim: int,
        embedding_dim: int,
        dropout: float,
        lr: float,
        weight_decay: float,
        margin: float,
        temperature: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.loss_name = loss_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.margin = margin
        self.temperature = temperature

        self.time_series_proj = nn.Sequential(
            nn.Linear(ts_input_dim, time_series_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(time_series_hidden_size, embedding_dim),
        )
        self.sector_proj = nn.Linear(sector_dim, sector_embedding_dim)
        self.final_proj = nn.Linear(embedding_dim + sector_embedding_dim, embedding_dim)

        self.text_encoder = AutoModel.from_pretrained(backbone_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embedding_dim)

    def get_ts_sector_embedding(
        self, time_series: torch.Tensor, sector: torch.Tensor
    ) -> torch.Tensor:
        batch_size = time_series.shape[0]
        flattened = time_series.reshape(batch_size, -1)
        ts_embedding = self.time_series_proj(flattened)
        sector_embedding = self.sector_proj(sector)
        combined = torch.cat([ts_embedding, sector_embedding], dim=-1)
        return self.final_proj(combined)

    def get_text_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = pick_text_embedding(outputs)
        return self.text_proj(pooled)

    def contrastive_loss(
        self, ts_embedding: torch.Tensor, text_embedding: torch.Tensor
    ) -> torch.Tensor:
        ts_norm = f.normalize(ts_embedding, dim=-1)
        text_norm = f.normalize(text_embedding, dim=-1)
        logits = torch.matmul(ts_norm, text_norm.t()) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_a = f.cross_entropy(logits, labels)
        loss_b = f.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_a + loss_b)

    def triplet_loss(
        self,
        anchor_embedding: torch.Tensor,
        positive_embedding: torch.Tensor,
        negative_embedding: torch.Tensor,
    ) -> torch.Tensor:
        positive_distance = f.pairwise_distance(
            anchor_embedding, positive_embedding, p=2
        )
        negative_distance = f.pairwise_distance(
            anchor_embedding, negative_embedding, p=2
        )
        return torch.relu(positive_distance - negative_distance + self.margin).mean()

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        anchor = batch["anchor_summary"]
        anchor_embedding = self.get_text_embedding(
            anchor["input_ids"], anchor["attention_mask"]
        )
        positive_embedding = self.get_ts_sector_embedding(
            batch["positive_time_series"], batch["positive_sector"]
        )

        if self.loss_name == "triplet":
            negative_embedding = self.get_ts_sector_embedding(
                batch["negative_time_series"], batch["negative_sector"]
            )
            loss = self.triplet_loss(
                anchor_embedding, positive_embedding, negative_embedding
            )
        elif self.loss_name == "contrastive":
            loss = self.contrastive_loss(positive_embedding, anchor_embedding)
        else:
            raise ValueError(f"Unsupported loss: {self.loss_name}")

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        anchor = batch["anchor_summary"]
        anchor_embedding = self.get_text_embedding(
            anchor["input_ids"], anchor["attention_mask"]
        )
        positive_embedding = self.get_ts_sector_embedding(
            batch["positive_time_series"], batch["positive_sector"]
        )

        if self.loss_name == "triplet":
            negative_embedding = self.get_ts_sector_embedding(
                batch["negative_time_series"], batch["negative_sector"]
            )
            loss = self.triplet_loss(
                anchor_embedding, positive_embedding, negative_embedding
            )
        else:
            loss = self.contrastive_loss(positive_embedding, anchor_embedding)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


def build_dataloaders(
    train_jsonl: Path,
    tokenizer: AutoTokenizer,
    loss_name: str,
    batch_size: int,
    val_split: float,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    rows = load_jsonl(train_jsonl)
    if not rows:
        raise RuntimeError(f"No training rows in {train_jsonl}")

    if len(rows) < 2:
        raise RuntimeError(
            f"Need at least 2 rows for train/validation split: {train_jsonl}"
        )

    if loss_name == "triplet":
        dataset: Dataset = TripletDataset(rows, tokenizer)
        collator = TripletCollator()
    elif loss_name == "contrastive":
        dataset = PositivePairDataset(rows, tokenizer)
        collator = PositivePairCollator()
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    val_size = max(1, int(round(len(dataset) * val_split)))
    train_size = len(dataset) - val_size
    if train_size < 1:
        train_size = len(dataset) - 1
        val_size = 1

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return train_loader, val_loader


def load_policy_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"fold", "Family Summary", "Document ID", "text_file_path"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    out = df.copy()
    out["Family Summary"] = out["Family Summary"].apply(clean_text)
    out["Document ID"] = out["Document ID"].astype(str)
    out["text_file_path"] = out["text_file_path"].astype(str)
    out = out[(out["Family Summary"] != "") & (out["Document ID"] != "")]
    out["fold"] = out["fold"].astype(int)
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
) -> pd.DataFrame:
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

    while True:
        payload = store.get(
            include=["documents", "metadatas"],
            limit=page_size,
            offset=offset,
        )
        page_documents = payload.get("documents", []) or []
        page_metadatas = payload.get("metadatas", []) or []

        if not page_documents:
            break

        documents.extend(page_documents)
        metadatas.extend(page_metadatas)

        if len(page_documents) < page_size:
            break
        offset += page_size

    if not documents or not metadatas:
        raise RuntimeError(
            f"No chunk entries found in vector DB collection `{resolved_collection}` at {vectordb_dir}."
        )

    fold_set = set(int(fold_id) for fold_id in chunk_fold_ids)
    rows: list[dict[str, str]] = []
    for chunk_text, metadata in zip(documents, metadatas):
        metadata = metadata or {}
        fold_value = metadata_get_int(metadata, ["fold", "Fold"])
        doc_id = metadata_get_str(
            metadata,
            ["document_id", "Document ID", "doc_id", "documentId"],
        )
        cleaned_chunk = clean_text(chunk_text)

        if fold_value is None or not doc_id or not cleaned_chunk:
            continue

        if fold_value not in fold_set:
            continue

        rows.append({"chunk_text": cleaned_chunk, "Document ID": doc_id})

    chunk_df = pd.DataFrame(rows)
    if chunk_df.empty:
        raise RuntimeError(
            "Chunk corpus from vector DB is empty for selected training folds. "
            "Check stored fold metadata and train/test split settings."
        )

    return chunk_df.drop_duplicates().reset_index(drop=True)


def embed_text_corpus(
    model: RetrievalDualEncoder,
    tokenizer: AutoTokenizer,
    corpus_df: pd.DataFrame,
    text_column: str,
    doc_id_column: str,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    summaries = corpus_df[text_column].astype(str).tolist()
    doc_ids = corpus_df[doc_id_column].astype(str).tolist()

    vectors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for idx in range(0, len(summaries), batch_size):
            batch_text = summaries[idx : idx + batch_size]
            tokenized = tokenizer(
                batch_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            emb = model.get_text_embedding(
                tokenized["input_ids"], tokenized["attention_mask"]
            )
            vectors.append(emb.detach().cpu().numpy())

    matrix = (
        np.vstack(vectors) if vectors else np.zeros((0, model.hparams.embedding_dim))
    )
    return matrix, doc_ids


def embed_query_ts(
    model: RetrievalDualEncoder, ts: list, sector: list, device: str
) -> np.ndarray:
    ts_tensor = torch.tensor(ts, dtype=torch.float32).unsqueeze(0).to(device)
    sector_tensor = torch.tensor(sector, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        emb = model.get_ts_sector_embedding(ts_tensor, sector_tensor)
    return emb.detach().cpu().numpy()


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


def ranked_doc_ids_for_query(
    query_vec: np.ndarray,
    corpus_matrix: np.ndarray,
    corpus_doc_ids: list[str],
) -> list[str]:
    if len(corpus_doc_ids) == 0:
        return []

    sims = cosine_similarity(query_vec, corpus_matrix)[0]
    agg_scores, agg_doc_ids = aggregate_by_doc_max(sims, corpus_doc_ids)
    order = np.argsort(-agg_scores)
    return [agg_doc_ids[i] for i in order]


def ranked_doc_ids_from_scores(
    scores: np.ndarray, corpus_doc_ids: list[str]
) -> list[str]:
    if len(corpus_doc_ids) == 0:
        return []

    agg_scores, agg_doc_ids = aggregate_by_doc_max(scores, corpus_doc_ids)
    order = np.argsort(-agg_scores)
    return [agg_doc_ids[i] for i in order]


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


def evaluate_retrieval(
    model: RetrievalDualEncoder,
    tokenizer: AutoTokenizer,
    chunk_corpus_df: pd.DataFrame,
    fold_id: int,
    window: int,
    backbone: str,
    loss_name: str,
    chunk_fold_ids: list[int],
    test_json: Path,
    k_values: list[int],
    device: str,
    batch_size: int,
    save_retrieval_traces: bool,
    retrieval_trace_top_k: int,
    retrieval_trace_output_path: Optional[Path],
) -> dict[str, float]:
    if chunk_corpus_df.empty:
        raise RuntimeError(f"Fold {fold_id}: empty retrieval chunk corpus")

    corpus_matrix, corpus_doc_ids = embed_text_corpus(
        model=model,
        tokenizer=tokenizer,
        corpus_df=chunk_corpus_df,
        text_column="chunk_text",
        doc_id_column="Document ID",
        device=device,
        batch_size=batch_size,
    )
    corpus_chunk_texts = chunk_corpus_df["chunk_text"].astype(str).tolist()

    with test_json.open("r", encoding="utf-8") as handle:
        test_rows = json.load(handle)

    if not test_rows:
        raise RuntimeError(f"No test rows in {test_json}")

    records = {f"hit@{k}": [] for k in k_values}
    records.update({f"precision@{k}": [] for k in k_values})
    records.update({f"ndcg@{k}": [] for k in k_values})
    records.update({f"mrr@{k}": [] for k in k_values})
    retrieval_trace_rows: list[dict] = []

    trace_top_k = retrieval_trace_top_k if retrieval_trace_top_k > 0 else max(k_values)

    skipped = 0
    for query_index, row in enumerate(test_rows):
        target = str(row.get("doc_id", ""))
        if not target:
            skipped += 1
            continue

        query_vec = embed_query_ts(
            model,
            ts=row["positive_time_series"],
            sector=row["positive_sector"],
            device=device,
        )
        sims = cosine_similarity(query_vec, corpus_matrix)[0]
        ranked = ranked_doc_ids_from_scores(sims, corpus_doc_ids)
        if not ranked:
            skipped += 1
            continue

        if save_retrieval_traces and retrieval_trace_output_path is not None:
            chunk_rank_order = np.argsort(-sims)
            max_rows = min(trace_top_k, len(chunk_rank_order))
            query_id = str(row.get("query_id", ""))
            for rank in range(max_rows):
                chunk_idx = int(chunk_rank_order[rank])
                retrieved_doc_id = str(corpus_doc_ids[chunk_idx])
                retrieval_trace_rows.append(
                    {
                        "fold": fold_id,
                        "window": window,
                        "backbone": backbone,
                        "loss": loss_name,
                        "query_index": query_index,
                        "query_id": query_id,
                        "target_doc_id": target,
                        "retrieved_rank": rank + 1,
                        "retrieved_doc_id": retrieved_doc_id,
                        "retrieved_score": float(sims[chunk_idx]),
                        "is_relevant_doc": float(retrieved_doc_id == target),
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

    metrics["queries_evaluated"] = float(max(len(test_rows) - skipped, 0))
    metrics["queries_total"] = float(len(test_rows))
    metrics["train_folds"] = ";".join(str(fold) for fold in sorted(chunk_fold_ids))
    metrics["chunk_folds"] = ";".join(str(fold) for fold in sorted(chunk_fold_ids))
    metrics["test_fold"] = str(fold_id)
    return metrics


def build_hparam_trials(
    args: argparse.Namespace, loss_name: str
) -> list[ModelHyperParams]:
    if not args.tune_hyperparams:
        return [
            ModelHyperParams(
                time_series_hidden_size=args.time_series_hidden_size,
                sector_embedding_dim=args.sector_embedding_dim,
                embedding_dim=args.embedding_dim,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                margin=args.margin,
                temperature=args.temperature,
            )
        ]

    hidden_sizes = parse_csv_ints(args.tune_time_series_hidden_size)
    sector_dims = parse_csv_ints(args.tune_sector_embedding_dim)
    embedding_dims = parse_csv_ints(args.tune_embedding_dim)
    dropouts = parse_csv_floats(args.tune_dropout)
    lrs = parse_csv_floats(args.tune_lr)
    weight_decays = parse_csv_floats(args.tune_weight_decay)

    if loss_name == "triplet":
        margins = parse_csv_floats(args.tune_margin)
        temperatures = [args.temperature]
    else:
        margins = [args.margin]
        temperatures = parse_csv_floats(args.tune_temperature)

    trials: list[ModelHyperParams] = []
    for (
        hidden_size,
        sector_dim,
        emb_dim,
        dropout,
        lr,
        weight_decay,
        margin,
        temperature,
    ) in product(
        hidden_sizes,
        sector_dims,
        embedding_dims,
        dropouts,
        lrs,
        weight_decays,
        margins,
        temperatures,
    ):
        trials.append(
            ModelHyperParams(
                time_series_hidden_size=int(hidden_size),
                sector_embedding_dim=int(sector_dim),
                embedding_dim=int(emb_dim),
                dropout=float(dropout),
                lr=float(lr),
                weight_decay=float(weight_decay),
                margin=float(margin),
                temperature=float(temperature),
            )
        )

    if args.tune_max_trials > 0 and len(trials) > args.tune_max_trials:
        rng = random.Random(args.seed)
        sampled_indices = sorted(rng.sample(range(len(trials)), args.tune_max_trials))
        trials = [trials[idx] for idx in sampled_indices]

    return trials


def train_one_model(
    run_dir: Path,
    stage_name: str,
    trial_id: int,
    fold_id: int,
    window: int,
    loss_name: str,
    backbone: str,
    ts_input_dim: int,
    sector_dim: int,
    hyperparams: ModelHyperParams,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: str,
    max_epochs: int,
    patience: int,
) -> tuple[str, float]:
    model = RetrievalDualEncoder(
        backbone_name=backbone,
        ts_input_dim=ts_input_dim,
        sector_dim=sector_dim,
        loss_name=loss_name,
        time_series_hidden_size=hyperparams.time_series_hidden_size,
        sector_embedding_dim=hyperparams.sector_embedding_dim,
        embedding_dim=hyperparams.embedding_dim,
        dropout=hyperparams.dropout,
        lr=hyperparams.lr,
        weight_decay=hyperparams.weight_decay,
        margin=hyperparams.margin,
        temperature=hyperparams.temperature,
    )

    logger = CSVLogger(
        save_dir=str(run_dir),
        name="logs",
        version=f"{stage_name}_trial_{trial_id:03d}",
    )
    ckpt_dir = run_dir / "checkpoints" / stage_name / f"trial_{trial_id:03d}"
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best",
        dirpath=str(ckpt_dir),
    )
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=patience)

    accelerator = "gpu" if device.startswith("cuda") else "cpu"
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint, early_stop],
        deterministic=args.deterministic,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)

    best_model_path = str(checkpoint.best_model_path)
    if not best_model_path:
        raise RuntimeError(
            f"Fold {fold_id}, window {window}: training produced no checkpoint for {stage_name}"
        )

    best_score = checkpoint.best_model_score
    if best_score is None:
        best_val_loss = float("inf")
    else:
        best_val_loss = float(best_score.detach().cpu().item())

    return best_model_path, best_val_loss


def discover_folds(model_input_dir: Path) -> list[int]:
    fold_ids: list[int] = []
    for path in sorted(model_input_dir.glob("fold_*")):
        if not path.is_dir():
            continue
        try:
            fold_ids.append(int(path.name.split("_")[1]))
        except Exception:
            continue
    if not fold_ids:
        raise RuntimeError(f"No fold directories found under {model_input_dir}")
    return sorted(set(fold_ids))


def resolve_eval_checkpoint(
    run_dir: Path,
    args: argparse.Namespace,
) -> str:
    if args.eval_checkpoint is not None:
        ckpt_path = args.eval_checkpoint
        if not ckpt_path.exists() or not ckpt_path.is_file():
            raise FileNotFoundError(
                f"--eval-checkpoint not found or not a file: {ckpt_path}"
            )
        return str(ckpt_path)

    direct_candidate = run_dir / "checkpoints" / "final" / "trial_001" / "best.ckpt"
    if direct_candidate.exists() and direct_candidate.is_file():
        return str(direct_candidate)

    pattern = str(run_dir / "checkpoints" / "final" / "trial_*" / "best.ckpt")
    matches = sorted(glob.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple checkpoint candidates found for eval-only. "
            "Provide --eval-checkpoint explicitly."
        )

    raise FileNotFoundError(
        "No checkpoint found for eval-only under run directory. "
        "Run training first or provide --eval-checkpoint."
    )


def hyperparams_from_dict(payload: dict) -> ModelHyperParams:
    return ModelHyperParams(
        time_series_hidden_size=int(payload["time_series_hidden_size"]),
        sector_embedding_dim=int(payload["sector_embedding_dim"]),
        embedding_dim=int(payload["embedding_dim"]),
        dropout=float(payload["dropout"]),
        lr=float(payload["lr"]),
        weight_decay=float(payload["weight_decay"]),
        margin=float(payload["margin"]),
        temperature=float(payload["temperature"]),
    )


def resolve_shared_tuned_hparams_dir(args: argparse.Namespace) -> Path:
    if args.shared_tuned_hparams_dir is not None:
        return args.shared_tuned_hparams_dir
    return args.output_dir / DEFAULT_SHARED_TUNED_HPARAMS_DIRNAME


def resolve_tuning_output_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / DEFAULT_TUNING_OUTPUT_DIRNAME


def resolve_tuning_run_dir(
    args: argparse.Namespace,
    fold_id: int,
    window: int,
    backbone: str,
    loss_name: str,
) -> Path:
    return (
        resolve_tuning_output_dir(args)
        / f"fold_{fold_id}"
        / f"window_{window}"
        / backbone.replace("/", "_")
        / loss_name
    )


def shared_tuned_hparams_path(
    shared_dir: Path,
    backbone: str,
    loss_name: str,
) -> Path:
    return shared_dir / backbone.replace("/", "_") / f"{loss_name}.json"


def load_best_tuned_hparams(run_dir: Path) -> ModelHyperParams:
    tuning_path = run_dir / "tuning_results.json"
    if not tuning_path.exists():
        raise FileNotFoundError(
            f"Missing tuning results at {tuning_path}. Run with --tune-only or --tune-hyperparams first."
        )

    with tuning_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    best_hparams = payload.get("best_hparams") if isinstance(payload, dict) else None
    if not isinstance(best_hparams, dict):
        raise ValueError(f"Invalid or missing best_hparams in {tuning_path}")

    return hyperparams_from_dict(best_hparams)


def load_shared_tuned_hparams(
    shared_dir: Path,
    backbone: str,
    loss_name: str,
) -> ModelHyperParams:
    tuning_path = shared_tuned_hparams_path(shared_dir, backbone, loss_name)
    if not tuning_path.exists():
        raise FileNotFoundError(
            f"Missing shared tuned hyperparameters at {tuning_path}. "
            "Run tuning with --shared-tuned-hparams first."
        )

    with tuning_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    best_hparams = payload.get("best_hparams") if isinstance(payload, dict) else None
    if not isinstance(best_hparams, dict):
        raise ValueError(f"Invalid or missing best_hparams in {tuning_path}")

    return hyperparams_from_dict(best_hparams)


def save_shared_tuned_hparams(
    shared_dir: Path,
    backbone: str,
    loss_name: str,
    fold_id: int,
    window: int,
    best_tune_val_loss: float,
    best_hparams: ModelHyperParams,
    trial_count: int,
) -> Path:
    target_path = shared_tuned_hparams_path(shared_dir, backbone, loss_name)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "scope": "shared_by_backbone_and_loss",
        "backbone": backbone,
        "loss": loss_name,
        "source_fold": fold_id,
        "source_window": window,
        "best_val_loss": best_tune_val_loss,
        "trial_count": trial_count,
        "best_hparams": best_hparams.__dict__,
    }
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return target_path


def train_and_eval_one(
    fold_id: int,
    chunk_fold_ids: list[int],
    window: int,
    backbone: str,
    loss_name: str,
    args: argparse.Namespace,
    device: str,
) -> Optional[dict]:
    run_dir = (
        args.output_dir
        / f"fold_{fold_id}"
        / f"window_{window}"
        / backbone.replace("/", "_")
        / loss_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    tuning_run_dir = resolve_tuning_run_dir(
        args=args,
        fold_id=fold_id,
        window=window,
        backbone=backbone,
        loss_name=loss_name,
    )

    train_jsonl = (
        args.model_input_dir / f"fold_{fold_id}" / f"window_{window}" / "train.jsonl"
    )
    test_json = (
        args.model_input_dir / f"fold_{fold_id}" / f"window_{window}" / "test.json"
    )
    if not train_jsonl.exists():
        raise FileNotFoundError(
            f"Missing fold/window training input: {train_jsonl}. "
            "Run scripts/2_data_preparation_and_time_series_scaling.py first."
        )
    if not args.train_only and not test_json.exists():
        raise FileNotFoundError(
            f"Missing fold/window test input: {test_json}. "
            "Run scripts/2_data_preparation_and_time_series_scaling.py first."
        )

    ts_input_dim, sector_dim = detect_input_dims(train_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    train_loader, val_loader = build_dataloaders(
        train_jsonl=train_jsonl,
        tokenizer=tokenizer,
        loss_name=loss_name,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed + fold_id + window,
        num_workers=args.num_workers,
    )

    if args.eval_only and args.tune_hyperparams:
        raise ValueError("--eval-only cannot be combined with --tune-hyperparams")
    if args.eval_only and args.tune_only:
        raise ValueError("--eval-only cannot be combined with --tune-only")
    if args.eval_only and args.train_only:
        raise ValueError("--eval-only cannot be combined with --train-only")

    trial_hparams = build_hparam_trials(args, loss_name)
    tuning_records: list[dict] = []
    best_trial_hparams = trial_hparams[0]
    hparams_source = "default"
    best_tune_val_loss = float("inf")
    ran_tuning = False

    if args.tune_hyperparams or args.tune_only:
        ran_tuning = True
        hparams_source = "tuned_current_run"
        print(
            f"[TUNING] Start fold={fold_id}, window={window}, backbone={backbone}, "
            f"loss={loss_name}, trials={len(trial_hparams)}"
        )
        for trial_idx, hparams in enumerate(trial_hparams, start=1):
            print(
                f"[TUNING] Trial {trial_idx}/{len(trial_hparams)} "
                f"hparams={hparams.__dict__}"
            )
            trial_ckpt_path, trial_val_loss = train_one_model(
                run_dir=tuning_run_dir,
                stage_name="tuning",
                trial_id=trial_idx,
                fold_id=fold_id,
                window=window,
                loss_name=loss_name,
                backbone=backbone,
                ts_input_dim=ts_input_dim,
                sector_dim=sector_dim,
                hyperparams=hparams,
                train_loader=train_loader,
                val_loader=val_loader,
                args=args,
                device=device,
                max_epochs=args.tune_max_epochs,
                patience=args.tune_patience,
            )
            tuning_records.append(
                {
                    "trial": trial_idx,
                    "best_val_loss": trial_val_loss,
                    "checkpoint": trial_ckpt_path,
                    "hparams": hparams.__dict__,
                }
            )
            print(
                f"[TUNING] Trial {trial_idx} done: val_loss={trial_val_loss:.6f}, "
                f"checkpoint={trial_ckpt_path}"
            )
            if trial_val_loss < best_tune_val_loss:
                best_tune_val_loss = trial_val_loss
                best_trial_hparams = hparams
                print(
                    f"[TUNING] New best trial={trial_idx}, "
                    f"best_val_loss={best_tune_val_loss:.6f}"
                )

        tuning_results_path = tuning_run_dir / "tuning_results.json"
        tuning_results_path.parent.mkdir(parents=True, exist_ok=True)
        with tuning_results_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "fold": fold_id,
                    "window": window,
                    "backbone": backbone,
                    "loss": loss_name,
                    "trials": tuning_records,
                    "best_val_loss": best_tune_val_loss,
                    "best_hparams": best_trial_hparams.__dict__,
                },
                handle,
                indent=2,
            )
            print(f"[TUNING] Saved results: {tuning_results_path}")

        if args.shared_tuned_hparams:
            shared_dir = resolve_shared_tuned_hparams_dir(args)
            shared_path = save_shared_tuned_hparams(
                shared_dir=shared_dir,
                backbone=backbone,
                loss_name=loss_name,
                fold_id=fold_id,
                window=window,
                best_tune_val_loss=best_tune_val_loss,
                best_hparams=best_trial_hparams,
                trial_count=len(trial_hparams),
            )
            print(f"[TUNING] Saved shared tuned hyperparameters: {shared_path}")

    if args.tune_only:
        print(
            f"Tuning complete fold={fold_id}, window={window}, backbone={backbone}, loss={loss_name}, "
            f"best_val_loss={best_tune_val_loss:.6f}"
        )
        return None

    if args.eval_only:
        final_ckpt_path = resolve_eval_checkpoint(run_dir=run_dir, args=args)
        final_val_loss = float("nan")
    else:
        if args.use_tuned_hparams and not ran_tuning:
            if args.shared_tuned_hparams:
                shared_dir = resolve_shared_tuned_hparams_dir(args)
                best_trial_hparams = load_shared_tuned_hparams(
                    shared_dir=shared_dir,
                    backbone=backbone,
                    loss_name=loss_name,
                )
                hparams_source = "shared_tuned"
            else:
                best_trial_hparams = load_best_tuned_hparams(tuning_run_dir)
                hparams_source = "per_run_tuned"
            best_tune_val_loss = float("nan")
            print(
                f"[TRAIN] Loaded tuned hyperparameters for fold={fold_id}, window={window}, "
                f"backbone={backbone}, loss={loss_name}: {best_trial_hparams.__dict__}"
            )
        else:
            if not ran_tuning:
                best_trial_hparams = trial_hparams[0]

        final_ckpt_path, final_val_loss = train_one_model(
            run_dir=run_dir,
            stage_name="final",
            trial_id=1,
            fold_id=fold_id,
            window=window,
            loss_name=loss_name,
            backbone=backbone,
            ts_input_dim=ts_input_dim,
            sector_dim=sector_dim,
            hyperparams=best_trial_hparams,
            train_loader=train_loader,
            val_loader=val_loader,
            args=args,
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )

    if args.train_only:
        result = {
            "fold": fold_id,
            "train_folds": ";".join(str(fold) for fold in sorted(chunk_fold_ids)),
            "chunk_folds": ";".join(str(fold) for fold in sorted(chunk_fold_ids)),
            "window": window,
            "backbone": backbone,
            "loss": loss_name,
            "best_checkpoint": final_ckpt_path,
            "final_val_loss": final_val_loss,
            "train_rows": len(train_loader.dataset),
            "val_rows": len(val_loader.dataset),
            "used_hparams": best_trial_hparams.__dict__,
            "used_hparams_source": hparams_source,
            "eval_only": bool(args.eval_only),
            "train_only": bool(args.train_only),
            "tune_only": bool(args.tune_only),
            "use_tuned_hparams": bool(args.use_tuned_hparams),
            "hyperparameter_tuning": bool(args.tune_hyperparams),
            "shared_tuned_hparams": bool(args.shared_tuned_hparams),
            "tuning_trials": len(trial_hparams) if ran_tuning else 0,
            "tune_best_val_loss": (
                best_tune_val_loss
                if ran_tuning
                else (float("nan") if args.eval_only else final_val_loss)
            ),
        }

        with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

        return result

    best_model = RetrievalDualEncoder.load_from_checkpoint(final_ckpt_path)
    best_model.to(device)

    chunk_corpus_df = load_chunk_corpus_from_vectordb(
        vectordb_dir=args.chunk_vectordb_dir,
        collection_name=args.chunk_vectordb_collection,
        chunk_fold_ids=chunk_fold_ids,
    )

    k_values = parse_csv_ints(args.k_values)
    metrics = evaluate_retrieval(
        model=best_model,
        tokenizer=tokenizer,
        chunk_corpus_df=chunk_corpus_df,
        fold_id=fold_id,
        window=window,
        backbone=backbone,
        loss_name=loss_name,
        chunk_fold_ids=chunk_fold_ids,
        test_json=test_json,
        k_values=k_values,
        device=device,
        batch_size=args.batch_size,
        save_retrieval_traces=args.save_retrieval_traces,
        retrieval_trace_top_k=args.retrieval_trace_top_k,
        retrieval_trace_output_path=(run_dir / "retrieval_traces.csv"),
    )

    result = {
        "fold": fold_id,
        "train_folds": ";".join(str(fold) for fold in sorted(chunk_fold_ids)),
        "chunk_folds": ";".join(str(fold) for fold in sorted(chunk_fold_ids)),
        "window": window,
        "backbone": backbone,
        "loss": loss_name,
        "best_checkpoint": final_ckpt_path,
        "final_val_loss": final_val_loss,
        "train_rows": len(train_loader.dataset),
        "val_rows": len(val_loader.dataset),
        "used_hparams": best_trial_hparams.__dict__,
        "used_hparams_source": hparams_source,
        "eval_only": bool(args.eval_only),
        "train_only": bool(args.train_only),
        "tune_only": bool(args.tune_only),
        "use_tuned_hparams": bool(args.use_tuned_hparams),
        "hyperparameter_tuning": bool(args.tune_hyperparams),
        "shared_tuned_hparams": bool(args.shared_tuned_hparams),
        "tuning_trials": len(trial_hparams) if ran_tuning else 0,
        "tune_best_val_loss": (
            best_tune_val_loss
            if ran_tuning
            else (float("nan") if args.eval_only else final_val_loss)
        ),
    }
    result.update(metrics)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    return result


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
        results_df.groupby(["window", "backbone", "loss"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values(["window", "backbone", "loss"])
    )
    grouped.to_csv(output_dir / "summary_mean_metrics.csv", index=False)


def compute_significance(
    results_df: pd.DataFrame, args: argparse.Namespace
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

    baseline_mask = (
        (results_df["backbone"] == args.baseline_backbone)
        & (results_df["loss"] == args.baseline_loss)
        & (results_df["window"] == args.baseline_window)
    )
    baseline = results_df[baseline_mask].copy()
    if baseline.empty:
        raise ValueError(
            "Baseline combination has no results. "
            "Check --baseline-backbone, --baseline-loss, --baseline-window."
        )

    stat_rows: list[dict] = []

    compared = results_df[["window", "backbone", "loss"]].drop_duplicates()
    for _, cfg in compared.iterrows():
        window = int(cfg["window"])
        backbone = str(cfg["backbone"])
        loss_name = str(cfg["loss"])

        if (
            window == args.baseline_window
            and backbone == args.baseline_backbone
            and loss_name == args.baseline_loss
        ):
            continue

        subset = results_df[
            (results_df["window"] == window)
            & (results_df["backbone"] == backbone)
            & (results_df["loss"] == loss_name)
        ].copy()

        merged = baseline.merge(
            subset,
            on="fold",
            suffixes=("_base", "_cand"),
            how="inner",
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
                    "baseline_backbone": args.baseline_backbone,
                    "baseline_loss": args.baseline_loss,
                    "baseline_window": args.baseline_window,
                    "candidate_backbone": backbone,
                    "candidate_loss": loss_name,
                    "candidate_window": window,
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


def main() -> None:
    args = parse_args()
    setup_reproducibility(seed=args.seed, deterministic=args.deterministic)

    if args.tune_only and not args.tune_hyperparams:
        raise ValueError("--tune-only requires --tune-hyperparams")
    if args.tune_only and args.eval_only:
        raise ValueError("--tune-only cannot be combined with --eval-only")
    if args.tune_only and args.train_only:
        raise ValueError("--tune-only cannot be combined with --train-only")
    if args.use_tuned_hparams and args.tune_hyperparams:
        raise ValueError(
            "--use-tuned-hparams cannot be combined with --tune-hyperparams"
        )
    if args.train_only and args.eval_only:
        raise ValueError("--train-only cannot be combined with --eval-only")
    if args.shared_tuned_hparams_dir is not None and not args.shared_tuned_hparams:
        raise ValueError("--shared-tuned-hparams-dir requires --shared-tuned-hparams")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.retrieval_trace_top_k < 0:
        raise ValueError("--retrieval-trace-top-k must be >= 0")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(
        f"[RUNTIME] device={device}, deterministic={args.deterministic}, "
        f"num_workers={args.num_workers}, tune={args.tune_hyperparams}, "
        f"tune_only={args.tune_only}, train_only={args.train_only}, eval_only={args.eval_only}"
    )

    windows = parse_csv_ints(args.windows)
    backbones = parse_csv_strs(args.backbones)
    losses = parse_csv_strs(args.losses)
    k_values = parse_csv_ints(args.k_values)
    if any(k <= 0 for k in k_values):
        raise ValueError("All k-values must be positive")

    if args.tune_hyperparams and args.shared_tuned_hparams:
        if args.fold is None:
            raise ValueError(
                "Shared tuned hyperparameters should be generated from one source fold. "
                "Set --fold <id> when using --tune-hyperparams --shared-tuned-hparams."
            )
        if len(windows) != 1:
            raise ValueError(
                "Shared tuned hyperparameters should be generated from one source window. "
                "Set exactly one value in --windows when using --tune-hyperparams --shared-tuned-hparams."
            )

    policy_df = load_policy_df(args.policy_input)

    available_fold_ids = discover_folds(args.model_input_dir)
    if args.fold is None:
        eval_fold_ids = available_fold_ids
    else:
        if args.fold not in available_fold_ids:
            raise ValueError(
                f"Fold {args.fold} not found in model input directory: {available_fold_ids}"
            )
        eval_fold_ids = [args.fold]

    all_results: list[dict] = []

    for fold_id in eval_fold_ids:
        chunk_fold_ids = [fold_id]

        for window in windows:
            for backbone in backbones:
                for loss_name in losses:
                    result = train_and_eval_one(
                        fold_id=fold_id,
                        chunk_fold_ids=chunk_fold_ids,
                        window=window,
                        backbone=backbone,
                        loss_name=loss_name,
                        args=args,
                        device=device,
                    )
                    if result is None:
                        continue
                    all_results.append(result)
                    print(
                        f"Finished fold={fold_id}, window={window}, backbone={backbone}, "
                        f"loss={loss_name}, hit@5={result.get('hit@5', 0):.4f}, "
                        f"ndcg@5={result.get('ndcg@5', 0):.4f}, "
                        f"precision@5={result.get('precision@5', 0):.4f}"
                    )

    if args.tune_only:
        print("Tuning-only run completed. Skipping aggregate train/eval reports.")
        return

    if args.train_only:
        training_results_path = args.output_dir / "all_fold_training_results.csv"
        pd.DataFrame(all_results).to_csv(training_results_path, index=False)
        print(f"Saved training-only results: {training_results_path}")
        print(
            "Train-only run completed. Skipping retrieval evaluation aggregate reports."
        )
        return

    results_df = pd.DataFrame(all_results)
    results_path = args.output_dir / "all_fold_results.csv"
    results_df.to_csv(results_path, index=False)

    summarize_fold_metrics(results_df, args.output_dir)

    significance_df = compute_significance(results_df, args)
    significance_path = args.output_dir / "paired_significance_tests.csv"
    significance_df.to_csv(significance_path, index=False)

    metadata = {
        "model_type": "siamese-style dual-encoder",
        "seed": args.seed,
        "deterministic": args.deterministic,
        "folds": eval_fold_ids,
        "available_folds": available_fold_ids,
        "fold_protocol": "use each fold/window prepared train and test files from the same fold directory",
        "windows": windows,
        "backbones": backbones,
        "losses": losses,
        "k_values": k_values,
        "max_epochs": args.max_epochs,
        "chunk_retrieval": {
            "enabled": True,
            "source": "persisted_vector_db",
            "vectordb_dir": str(args.chunk_vectordb_dir),
            "vectordb_collection": args.chunk_vectordb_collection,
            "chunk_fold_filter": "current_fold_only",
            "doc_score_aggregation": "max score across chunks per document",
            "save_retrieval_traces": bool(args.save_retrieval_traces),
            "retrieval_trace_top_k": int(args.retrieval_trace_top_k),
        },
        "hyperparameter_tuning": {
            "enabled": bool(args.tune_hyperparams),
            "tune_only": bool(args.tune_only),
            "use_tuned_hparams": bool(args.use_tuned_hparams),
            "tuning_output_dir": str(resolve_tuning_output_dir(args)),
            "shared_tuned_hparams": bool(args.shared_tuned_hparams),
            "shared_tuned_hparams_dir": (
                str(resolve_shared_tuned_hparams_dir(args))
                if args.shared_tuned_hparams
                else "disabled"
            ),
            "tune_max_epochs": args.tune_max_epochs,
            "tune_patience": args.tune_patience,
            "tune_max_trials": args.tune_max_trials,
        },
        "eval_only": {
            "enabled": bool(args.eval_only),
            "eval_checkpoint": (
                str(args.eval_checkpoint) if args.eval_checkpoint else "auto"
            ),
        },
        "policy_retrieval_note": (
            "Hit@k is complemented with Precision@k, NDCG@k, and MRR@k. "
            "For policy retrieval, NDCG@k emphasizes ranking actionable policy clauses "
            "near the top rather than only counting whether a relevant item appears anywhere in top-k."
        ),
        "results_csv": str(results_path),
        "significance_csv": str(significance_path),
    }

    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved results: {results_path}")
    print(f"Saved significance tests: {significance_path}")


if __name__ == "__main__":
    main()
