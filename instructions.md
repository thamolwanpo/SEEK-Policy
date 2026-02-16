# File Instructions Log

This document tracks usage instructions and operational notes for each script/file.
Use one section per file so this can scale as more files are added.

## File: scripts/0_split_test_set.py

### Purpose
Creates leakage-safe grouped K-fold train/test splits for POLiMATCH by
keeping all versions of each law family in the same fold.

### Inputs
- `data/csv/all_data_en.csv`
- `data/csv/region.csv`

### Outputs
- `data/csv/group_kfold_assignments.csv` (document rows + fold label)
- `data/csv/kfold/fold_<i>_train.csv` (per-fold train set)
- `data/csv/kfold/fold_<i>_test.csv` (per-fold test set)

### How to run
```bash
python scripts/0_split_test_set.py
```

Optional arguments:
- `--input <path>`: input policy CSV (default: `data/csv/all_data_en.csv`)
- `--region <path>`: region lookup CSV (default: `data/csv/region.csv`)
- `--output-dir <path>`: output folder (default: `data/csv`)
- `--n-folds <int>`: number of grouped CV folds (default: `5`)
- `--seed <int>`: random seed for deterministic fold assignment (default: `42`)
- `--kfold-dir-name <str>`: subfolder name for per-fold files (default: `kfold`)

Example:
```bash
python scripts/0_split_test_set.py --n-folds 5 --seed 42
```

### Split rules (must hold)
- K-fold assignment is group-based: all versions from the same family share one fold.
- For each fold, train/test are disjoint by family group (no leakage).
- Example guarantee: if a law family has a 2008 and a 2018 version, both versions are always in the same fold and can never be split between train and test.

### Validation checks in script output
- Global document/family counts
- Per-fold train/test sizes
- Per-fold family overlap count (must be `0`)

### Notes
- If a fold becomes empty, reduce `--n-folds`.
- If `Family ID` is missing, the script falls back to:
  `Internal Family ID` → `Family Title` → `Document ID` → `doc_id`.

## File: scripts/0_visualize_data_distribution.py

### Purpose
Generates distribution visualizations for K-fold train/test datasets (no benchmark set).

### Inputs
- `data/csv/all_data_en.csv`
- `data/csv/region.csv`
- `data/csv/kfold/fold_<i>_train.csv`
- `data/csv/kfold/fold_<i>_test.csv`

### Outputs
- `figures/kfold_summary/region_distribution_all_folds.svg`
- `figures/kfold_summary/country_distribution_all_folds.svg` (when map dataset is available)
- `figures/kfold_summary/sector_distribution_all_folds.svg`

### How to run
```bash
python scripts/0_visualize_data_distribution.py
```

Optional arguments:
- `--fold <int>`: visualize only one fold; if omitted, the script loops through all available folds and still saves consolidated summary figures.

### Validation
- Confirms train/test are loaded from `data/csv/kfold/fold_<i>_train.csv` and `fold_<i>_test.csv`.
- Produces one consolidated region figure across folds.
- Produces one consolidated sector figure across folds.
- Produces one consolidated country-map figure across folds when map data is available.

### Notes
- If selected fold files are missing, the script raises an error and prompts to run `scripts/0_split_test_set.py` first.
- For headless environments, run with `MPLBACKEND=Agg`.

## File: scripts/1_time_series_data_preprocessing.py

### Purpose
Preprocesses OWID time-series data in a leakage-safe way for grouped K-fold evaluation.
Feature filtering is fit only on training countries for each fold, then applied to both train and test countries in that fold.

### Inputs
- `data/owid/` (recursive OWID table folders with `*.csv` + `*.meta.json`)
- `data/csv/group_kfold_assignments.csv` (must include `Geography` and `fold`)

### Outputs
- `data/time_series/kfold/fold_<i>/train/<country>.csv`
- `data/time_series/kfold/fold_<i>/test/<country>.csv`
- `data/time_series/kfold/fold_<i>/manifest.json`
- `data/time_series/kfold/preprocessing_summary.json`

### How to run
```bash
python scripts/1_time_series_data_preprocessing.py
```

Optional arguments:
- `--owid-dir <path>`: OWID root directory (default: `data/owid`)
- `--policy-input <path>`: grouped fold assignment CSV (default: `data/csv/group_kfold_assignments.csv`)
- `--output-dir <path>`: output root (default: `data/time_series`)
- `--start-year <int>` / `--end-year <int>`: year window (defaults: `1970` to `2024`)
- `--missing-threshold <float>`: max missing share for feature retention (default: `0.2`)
- `--correlation-threshold <float>`: max pairwise absolute correlation before dropping a redundant feature (default: `0.7`)
- `--fold <int>`: process one fold only; otherwise all folds are processed.

Example:
```bash
python scripts/1_time_series_data_preprocessing.py --fold 0
```

### Leakage controls (must hold)
- Feature selection (missingness, constant filter, correlation filter) is fit on train-fold countries only.
- Test-fold countries never influence feature-retention decisions.
- Per-fold outputs are separated into `train/` and `test/` directories.

### Validation checks in script output
- Train/test country counts per fold
- Selected table and feature counts per fold
- Output row count and summary manifest path

### Notes
- Run `scripts/0_split_test_set.py` first to generate `group_kfold_assignments.csv`.
- The script expects `Geography` values in policy input to match OWID `country` names.

## File: scripts/2_data_preparation_and_time_series_scaling.py

### Purpose
Builds model-ready policy/time-series datasets per fold, with leakage-safe scaling and multi-window ablation outputs.

### Inputs
- `data/csv/group_kfold_assignments.csv` (must include `fold`, `Family Summary`, `Sector`, `Geography`, `Last event in timeline`)
- `data/time_series/kfold/fold_<i>/train/<country>.csv`
- `data/time_series/kfold/fold_<i>/test/<country>.csv`

### Outputs
- `data/model_input/kfold/fold_<i>/scaled_train_time_series.csv`
- `data/model_input/kfold/fold_<i>/scaled_test_time_series.csv`
- `data/model_input/kfold/fold_<i>/scaler.pkl`
- `data/model_input/kfold/fold_<i>/window_<w>/train.jsonl`
- `data/model_input/kfold/fold_<i>/window_<w>/test.json`
- `data/model_input/kfold/fold_<i>/manifest.json`
- `data/model_input/kfold/preparation_summary.json`

### How to run
```bash
python scripts/2_data_preparation_and_time_series_scaling.py
```

Example (single fold + explicit ablation windows):
```bash
python scripts/2_data_preparation_and_time_series_scaling.py --fold 0 --windows 1,2,5,10
```

### Important options
- `--policy-input <path>`: grouped fold assignment CSV (default: `data/csv/group_kfold_assignments.csv`)
- `--time-series-dir <path>`: fold time-series root (default: `data/time_series/kfold`)
- `--output-dir <path>`: output root (default: `data/model_input/kfold`)
- `--windows <csv>`: history windows for ablation (default: `1,2,5,10`)
- `--negative-samples <int>`: negatives per positive anchor for train JSONL (default: `1`)
- `--fold <int>`: process one fold only; if omitted all folds are processed
- `--seed <int>`: deterministic negative sampling seed (default: `42`)

### Leakage controls (must hold)
- Time-series scaler is fit using train-fold country rows only.
- The fitted scaler is applied to both train and test country rows.
- Fold outputs are isolated under `fold_<i>/` and ablation windows are isolated under `window_<w>/`.

### Validation checks in script output
- Per-fold train/test policy row counts
- Per-window train pair and test row counts
- Saved summary manifest location

### Notes
- Run `scripts/1_time_series_data_preprocessing.py` first so fold-specific country files exist.
- Window sizes larger than available history are left-padded with zeros.

## File: scripts/3_build_chunk_vectordb.py

### Purpose
Builds a persisted Chroma chunk vector database from policy text files using OpenAI embeddings.
Each chunk stores fold/document metadata for fold-aware retrieval and evaluation.

### Inputs
- `data/csv/group_kfold_assignments.csv` (must include `fold`, `Document ID`, `text_file_path`, `Family Summary`)
- Text files referenced by `text_file_path`
- `.env` with `OPENAI_API_KEY`

### Outputs
- `data/vectorstore/policy_chunks_chroma/` (persisted Chroma collection)
- `data/vectorstore/policy_chunks_chroma/manifest.json`

### How to run
```bash
python scripts/3_build_chunk_vectordb.py
```

Example (full rebuild + custom embedding batch size):
```bash
python scripts/3_build_chunk_vectordb.py --rebuild --embedding-batch-size 128
```

Example (quick smoke run on a subset):
```bash
python scripts/3_build_chunk_vectordb.py --max-docs 100 --rebuild
```

### Important options
- `--policy-input <path>`: grouped fold assignment CSV (default: `data/csv/group_kfold_assignments.csv`)
- `--vectordb-dir <path>`: output vector DB directory (default: `data/vectorstore/policy_chunks_chroma`)
- `--collection <str>`: Chroma collection name (default: `policy_chunks_openai`)
- `--text-path-base-dir <path>`: base directory for resolving relative `text_file_path` values (default: `data/csv`)
- `--chunk-size <int>`: character chunk size (default: `1500`)
- `--chunk-overlap <int>`: chunk overlap characters (default: `200`)
- `--embedding-model <str>`: OpenAI embedding model (default: `text-embedding-3-small`)
- `--embedding-batch-size <int>`: chunk embedding API batch size (default: `128`)
- `--rebuild`: delete and rebuild existing vector DB directory
- `--max-docs <int>`: optional cap on unique documents processed (`0` = all)

### Validation checks in script output
- Input CSV load row count and unique document count
- Document chunking progress and skipped document count
- Embedding progress (`Chunks left: ...`) and periodic ingest heartbeat
- Final vector DB path, collection name, chunk count, and manifest path

### Notes
- Relative `text_file_path` values are resolved against `--text-path-base-dir`, current working directory, and a `data/` fallback for `../` paths.
- If no valid text is found after cleaning/normalization, the script raises an error (`No chunk documents were built...`).
- `manifest.json` records chunk settings, model, folds, and ingested/skipped counts for reproducibility.

## File: scripts/4_train_siamese.py

### Purpose
Trains and evaluates retrieval models across grouped folds with configurable
window/backbone/loss experiment combinations.
Each fold/window run consumes `train.jsonl` and `test.json` from the same fold directory.

### Inputs
- `data/csv/group_kfold_assignments.csv`
- `data/model_input/kfold/fold_<i>/window_<w>/train.jsonl`
- `data/model_input/kfold/fold_<i>/window_<w>/test.json`
- `data/vectorstore/policy_chunks_chroma/` (persisted chunk DB from `scripts/3_build_chunk_vectordb.py`)

### Outputs
- Per-run logs and checkpoints:
  - `results/retrieval_experiments/fold_<i>/window_<w>/<backbone>/<loss>/`
  - optional per-query retrieval traces:
    - `results/retrieval_experiments/fold_<i>/window_<w>/<backbone>/<loss>/retrieval_traces.csv`
- Aggregated reports:
  - `results/retrieval_experiments/all_fold_results.csv`
  - `results/retrieval_experiments/summary_mean_metrics.csv`
  - `results/retrieval_experiments/paired_significance_tests.csv`
  - `results/retrieval_experiments/run_metadata.json`

### How to run
```bash
python scripts/4_train_siamese.py --max-epochs 10
```

Example (single fold):
```bash
python scripts/4_train_siamese.py --fold 0 --windows 1 --max-epochs 10
```

Example (hyperparameter tuning):
```bash
python scripts/4_train_siamese.py \
  --fold 0 \
  --windows 1 \
  --backbones climatebert/distilroberta-base-climate-f \
  --losses triplet,contrastive \
  --tune-hyperparams \
  --tune-max-trials 12 \
  --tune-max-epochs 3 \
  --tune-patience 2
```

Example (quick tuning only, recommended when many experiments are queued):
```bash
python scripts/4_train_siamese.py \
  --fold 0 \
  --windows 1 \
  --backbones climatebert/distilroberta-base-climate-f \
  --losses triplet \
  --tune-hyperparams \
  --tune-only \
  --tune-max-trials 6 \
  --tune-max-epochs 2 \
  --tune-patience 1
```

Example (train+eval later using previously tuned hyperparameters):
```bash
python scripts/4_train_siamese.py \
  --fold 0 \
  --windows 1 \
  --backbones climatebert/distilroberta-base-climate-f \
  --losses triplet \
  --use-tuned-hparams \
  --max-epochs 8
```

Example (tune once, then reuse across all folds/windows):
```bash
# pass 1: tune once using one source fold/window
python scripts/4_train_siamese.py \
  --fold 0 \
  --windows 5 \
  --backbones climatebert/distilroberta-base-climate-f \
  --losses triplet,contrastive \
  --tune-hyperparams \
  --tune-only \
  --shared-tuned-hparams \
  --tune-max-trials 12 \
  --tune-max-epochs 3 \
  --tune-patience 2

# pass 2: run all folds/windows with shared tuned params
python scripts/4_train_siamese.py \
  --backbones climatebert/distilroberta-base-climate-f \
  --losses triplet,contrastive \
  --use-tuned-hparams \
  --shared-tuned-hparams \
  --max-epochs 10

# pass 2 (long run): same as above with 50 epochs
python scripts/4_train_siamese.py \
  --backbones climatebert/distilroberta-base-climate-f \
  --losses triplet,contrastive \
  --use-tuned-hparams \
  --shared-tuned-hparams \
  --max-epochs 50
```

Example (eval only, auto checkpoint):
```bash
python scripts/4_train_siamese.py \
  --fold 0 \
  --windows 1 \
  --backbones climatebert/distilroberta-base-climate-f \
  --losses triplet \
  --eval-only
```

Example (eval only, explicit checkpoint):
```bash
python scripts/4_train_siamese.py \
  --fold 0 \
  --windows 1 \
  --backbones climatebert/distilroberta-base-climate-f \
  --losses triplet \
  --eval-only \
  --eval-checkpoint results/retrieval_experiments/fold_0/window_1/climatebert_distilroberta-base-climate-f/triplet/checkpoints/final/trial_001/best.ckpt
```

Example (save per-query retrieved chunk traces):
```bash
python scripts/4_train_siamese.py \
  --fold 0 \
  --windows 1 \
  --backbones climatebert/distilroberta-base-climate-f \
  --losses triplet \
  --save-retrieval-traces \
  --retrieval-trace-top-k 10
```

### Important options
- `--windows <csv>`: default `1,2,5,10`
- `--backbones <csv>`: default
  `climatebert/distilroberta-base-climate-f,sentence-transformers/all-distilroberta-v1`
- `--losses <csv>`: default `triplet,contrastive`
- `--k-values <csv>`: ranking cutoffs for evaluation (default `1,5,10`)
- `--max-epochs <int>`: epochs per run (set > 3 for stable convergence tracking)
- `--chunk-vectordb-dir <path>`: persisted chunk DB directory (default `data/vectorstore/policy_chunks_chroma`)
- `--chunk-vectordb-collection <str>`: collection name; supports `auto`/`manifest` resolution from chunk DB manifest
- `--eval-only`: skip training and evaluate using an existing checkpoint
- `--eval-checkpoint <path>`: explicit checkpoint for eval-only mode
- `--save-retrieval-traces`: save ranked retrieved chunks per query to per-run `retrieval_traces.csv`
- `--retrieval-trace-top-k <int>`: number of chunk rows saved per query (`0` = use `max(--k-values)`)
- `--baseline-backbone`, `--baseline-loss`, `--baseline-window`: baseline for paired tests

Tuning options:
- `--tune-hyperparams`
- `--tune-only` (run only tuning and write `tuning_results.json`)
- `--use-tuned-hparams` (load best params from existing `tuning_results.json` for final training)
- `--shared-tuned-hparams` (save/load shared tuned params by backbone/loss for reuse across folds/windows)
- `--shared-tuned-hparams-dir <path>` (override default shared directory)
- default shared file layout: `results/retrieval_experiments/shared_tuned_hparams/<backbone>/<loss>.json`
- `--tune-max-epochs`, `--tune-patience`, `--tune-max-trials`
- `--tune-lr`, `--tune-weight-decay`
- `--tune-time-series-hidden-size`, `--tune-sector-embedding-dim`, `--tune-embedding-dim`
- `--tune-dropout`, `--tune-margin`, `--tune-temperature`

Runtime/reproducibility options:
- `--seed`, `--deterministic`, `--non-deterministic`, `--num-workers`, `--device`

### Eval-only checkpoint lookup order
1. `--eval-checkpoint` (if provided)
2. `<run_dir>/checkpoints/final/trial_001/best.ckpt`
3. Single match under `<run_dir>/checkpoints/final/trial_*/best.ckpt`

### Constraints
- `--eval-only` cannot be combined with `--tune-hyperparams`.
- `--tune-only` requires `--tune-hyperparams`.
- `--tune-only` cannot be combined with `--eval-only`.
- `--use-tuned-hparams` cannot be combined with `--tune-hyperparams`.
- `--shared-tuned-hparams-dir` requires `--shared-tuned-hparams`.
- `--tune-hyperparams --shared-tuned-hparams` requires one source fold (`--fold`) and one source window.

### Metrics and evaluation
- Reports `Hit@k`, `Precision@k`, `NDCG@k`, and `MRR@k` for each fold and experiment.
- Logs `train_loss` and `val_loss` by epoch via Lightning CSV logs for convergence tracking.
- Runs paired comparisons across fold scores (candidate vs baseline):
  - Paired t-test
  - Wilcoxon signed-rank test
  - Outputs p-values in `paired_significance_tests.csv`.

### Retrieval trace schema (when enabled)
- One row per query per retrieved chunk rank.
- Core columns: `fold`, `window`, `backbone`, `loss`, `query_index`, `query_id`, `target_doc_id`, `retrieved_rank`, `retrieved_doc_id`, `retrieved_score`, `is_relevant_doc`, `retrieved_chunk_text`.

### Why add NDCG/Precision for policy retrieval
- `Hit@k` alone can still rank useful policy clauses low in the top-k list.
- `NDCG@k` rewards higher-ranked relevant policies, aligning with practical use where
  top-ranked results must be actionable.
- `Precision@k` measures concentration of relevance in top positions and penalizes noisy rankings.

## File: scripts/4_summary_only_baselines.py

### Purpose
Runs two retrieval baselines that use only policy text summaries (no time-series features):
- Semantic retrieval with SentenceTransformer (`HumanSummaryBaseline`)
- Keyword retrieval with BM25 (`ClimatePolicyRadarBaseline`)

### Inputs
- `data/csv/group_kfold_assignments.csv` (must include `fold`, `Family Summary`, `Document ID`)

### Outputs
- `results/summary_only_baselines/all_fold_results.csv`
- `results/summary_only_baselines/summary_mean_metrics.csv`
- `results/summary_only_baselines/paired_significance_tests.csv`
- `results/summary_only_baselines/run_metadata.json`

### How to run
```bash
python scripts/4_summary_only_baselines.py
```

Example (single fold):
```bash
python scripts/4_summary_only_baselines.py --fold 0
```

### Important options
- `--policy-input <path>`: grouped fold assignment CSV (default: `data/csv/group_kfold_assignments.csv`)
- `--output-dir <path>`: output folder (default: `results/summary_only_baselines`)
- `--k-values <csv>`: ranking cutoffs (default: `1,5,10`)
- `--encoder <model_name>`: SentenceTransformer model id (default: `sentence-transformers/all-distilroberta-v1`)
- `--fold <int>`: evaluate one fold only; if omitted all folds are evaluated
- `--significance-baseline <str>`: baseline name used as reference in paired tests (default: `human_summary_semantic`)

### Metrics and evaluation
- Uses the same metrics as `scripts/4_train_siamese.py`:
  - `Hit@k`
  - `Precision@k`
  - `NDCG@k`
- Also computes fold-level paired significance tests across baselines:
  - paired t-test p-value
  - Wilcoxon signed-rank p-value
- For each fold:
  - Retrieval corpus = all rows where `fold != current_fold`
  - Queries/targets = rows where `fold == current_fold`
  - Query text is `Family Summary`
  - Target identifier is `Document ID`

### Dependencies
```bash
pip install sentence-transformers rank-bm25
```

## File: scripts/4_generate_role_agents_summary.py

### Purpose
Runs role-agent summary retrieval experiments with fold-aware evaluation and statistical testing.
It reuses the same evaluation concepts as `scripts/4_summary_only_baselines.py` and `scripts/4_train_siamese.py`.

### Inputs
- `data/csv/group_kfold_assignments.csv` (or another CSV with equivalent columns)
- Required columns (defaults):
  - fold: `fold`
  - document id: `Document ID`
  - human summary: `Family Summary`
  - role-agent query summary: `summarizer_v1` (can be generated by this script)

### Outputs
- `results/role_agent_summary_experiments/all_fold_results.csv`
- `results/role_agent_summary_experiments/summary_mean_metrics.csv`
- `results/role_agent_summary_experiments/paired_significance_tests.csv`
- `results/role_agent_summary_experiments/run_metadata.json`
- optional generated summary CSV when `--generate-role-summary` is used

### How to run
```bash
python scripts/4_generate_role_agents_summary.py --query-column summarizer_v1
```

Example (single fold):
```bash
python scripts/4_generate_role_agents_summary.py --fold 0 --query-column summarizer_v1
```

Example (generate summaries + evaluate):
```bash
python scripts/4_generate_role_agents_summary.py \
  --generate-role-summary \
  --source-column "Family Summary" \
  --query-column summarizer_v1
```

### Important options
- `--policy-input <path>`: input CSV (default: `data/csv/group_kfold_assignments.csv`)
- `--output-dir <path>`: output folder (default: `results/role_agent_summary_experiments`)
- `--encoder <model_name>`: SentenceTransformer model (default: `sentence-transformers/all-distilroberta-v1`)
- `--k-values <csv>`: ranking cutoffs (default: `1,5,10`)
- `--query-column <str>`: role-agent summary column to use as query text (default: `summarizer_v1`)
- `--generate-role-summary`: generate missing query summaries via OpenAI
- `--source-column <str>`: source text column for generation (default: `Family Summary`)
- `--openai-model <str>`: OpenAI model used for generation (default: `gpt-4o-mini`)
- `--significance-baseline <str>`: baseline method for paired tests (default: `role_agent_summary_semantic`)

### Metrics and evaluation
- Reports the same core retrieval metrics:
  - `Hit@k`
  - `Precision@k`
  - `NDCG@k`
- Computes fold-level paired significance tests between methods:
  - paired t-test p-value
  - Wilcoxon signed-rank p-value

### Environment
Create `.env` with:
```dotenv
OPENAI_API_KEY=
```

### Dependencies
```bash
pip install sentence-transformers rank-bm25 openai python-dotenv
```

## File: scripts/4_seekpolicy.py

### Purpose
Runs SEEK-Policy RAG summary retrieval experiments with the same evaluation concept as other retrieval scripts:
- optional generation of `RAG_v1_summary` via persisted Chroma + OpenAI
- fold-aware retrieval evaluation
- paired significance testing across folds

### Inputs
- `data/csv/group_kfold_assignments.csv` (or equivalent CSV)
- Required columns (defaults):
  - fold: `fold`
  - document id: `Document ID`
  - human summary: `Family Summary`
  - RAG summary query: `RAG_v1_summary` (can be generated by this script)

### Outputs
- `results/seekpolicy_experiments/all_fold_results.csv`
- `results/seekpolicy_experiments/summary_mean_metrics.csv`
- `results/seekpolicy_experiments/paired_significance_tests.csv`
- `results/seekpolicy_experiments/run_metadata.json`
- optional generated summary CSV when `--generate-rag-summary` is used

### How to run
```bash
python scripts/4_seekpolicy.py --query-column RAG_v1_summary
```

Example (single fold):
```bash
python scripts/4_seekpolicy.py --fold 0 --query-column RAG_v1_summary
```

Example (generate RAG summaries + evaluate):
```bash
python scripts/4_seekpolicy.py \
  --generate-rag-summary \
  --source-column "Family Summary" \
  --query-column RAG_v1_summary
```

### Important options
- `--policy-input <path>`: input CSV (default: `data/csv/group_kfold_assignments.csv`)
- `--output-dir <path>`: output folder (default: `results/seekpolicy_experiments`)
- `--encoder <model_name>`: SentenceTransformer model (default: `sentence-transformers/all-distilroberta-v1`)
- `--k-values <csv>`: ranking cutoffs (default: `1,5,10`)
- `--query-column <str>`: RAG summary query column (default: `RAG_v1_summary`)
- `--generate-rag-summary`: generate missing query summaries with persisted Chroma + OpenAI
- `--source-column <str>`: source text column for RAG generation (default: `Family Summary`)
- `--openai-model <str>`: OpenAI model for generation (default: `gpt-4o-mini`)
- `--rag-k <int>`: retrieved chunk count for generation (default: `5`)
- `--text-path-column <str>`: text file path column used to build/load Chroma chunks (default: `text_file_path`)
- `--chroma-dir <path>`: persisted Chroma directory (default: `data/vectorstore/seekpolicy_chroma`)
- `--rebuild-chroma`: rebuild Chroma index from policy text files before generation
- `--significance-baseline <str>`: baseline method for paired tests (default: `rag_summary_semantic`)

### Metrics and evaluation
- Reports:
  - `Hit@k`
  - `Precision@k`
  - `NDCG@k`
- Computes fold-level paired significance tests:
  - paired t-test p-value
  - Wilcoxon signed-rank p-value

### Environment
Create `.env` with:
```dotenv
OPENAI_API_KEY=
```

### Dependencies
```bash
pip install sentence-transformers rank-bm25 openai python-dotenv langchain-openai langchain-community langchain-chroma chromadb
```

---

## Template for next files

## File: <path/to/file>

### Purpose
- ...

### Inputs
- ...

### Outputs
- ...

### How to run
```bash
<command>
```

### Important options
- ...

### Validation
- ...

### Notes
- ...
