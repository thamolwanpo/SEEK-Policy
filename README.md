# SEEK-Policy

This README currently documents these scripts in `scripts/`:
- `0_split_test_set.py`
- `0_visualize_data_distribution.py`
- `1_time_series_data_preprocessing.py`
- `2_data_preparation_and_time_series_scaling.py`
- `3_build_chunk_vectordb.py`
- `4_train_siamese.py`
- `4_summary_only_baselines.py`
- `4_generate_role_agents_summary.py`
- `4_seekpolicy.py`

For full per-file operational notes (including `scripts/3_build_chunk_vectordb.py`), see `instructions.md`.

## 1) `scripts/0_split_test_set.py`
**Purpose**
- Creates leakage-safe grouped K-fold splits from policy data.
- Keeps all versions of the same law/family in the same fold.

**Main inputs (default)**
- `data/csv/all_data_en.csv`
- `data/csv/region.csv`

**Main outputs**
- `data/csv/group_kfold_assignments.csv` (original data + `fold` column)
- `data/csv/kfold/fold_{i}_train.csv`
- `data/csv/kfold/fold_{i}_test.csv`

**Key behavior**
- Builds a group key from fields such as `Family ID`, `Internal Family ID`, `Family Title`, `Document ID`.
- Randomly assigns groups to folds using a fixed seed (default `42`).
- Validates no train/test group overlap per fold to prevent leakage.

**Run example**
```bash
python scripts/0_split_test_set.py
```

Optional arguments:
- `--input`, `--region`, `--output-dir`, `--n-folds`, `--seed`, `--kfold-dir-name`

---

## 2) `scripts/0_visualize_data_distribution.py`
**Purpose**
- Visualizes train/test data distribution across grouped K-fold splits.

**Input source**
- Reads fold files from `data/csv/kfold/`.
- Expects files created by `0_split_test_set.py`.

**Main outputs**
- `figures/kfold_summary/region_distribution_all_folds.svg`
- `figures/kfold_summary/sector_distribution_all_folds.svg`
- `figures/kfold_summary/country_distribution_all_folds.svg` (only if `geopandas` and map data are available)

**Key behavior**
- Region chart: compares train/test % by region for each fold.
- Sector chart: splits multi-sector entries, removes `Other`, and compares percentages.
- Country map chart: train/test country coverage per fold (skips gracefully if `geopandas` is unavailable).

**Run examples**
```bash
# all folds
python scripts/0_visualize_data_distribution.py

# specific fold only
python scripts/0_visualize_data_distribution.py --fold 0
```

---

## 3) `scripts/1_time_series_data_preprocessing.py`
**Purpose**
- Preprocesses OWID time-series features in a leakage-safe way for each fold.
- Feature selection is fit on training countries only, then applied to both train/test countries.

**Main inputs (default)**
- OWID source directory: `data/owid/`
- Policy fold assignments: `data/csv/group_kfold_assignments.csv`

**Main outputs**
- Per-fold country-level time series:
  - `data/time_series/kfold/fold_{i}/train/*.csv`
  - `data/time_series/kfold/fold_{i}/test/*.csv`
- Fold manifest:
  - `data/time_series/kfold/fold_{i}/manifest.json`
- Overall summary:
  - `data/time_series/kfold/preprocessing_summary.json`

**Key behavior**
- Discovers OWID tables from `*.meta.json` + matching `*.csv`.
- Standardizes to `country` and `year`, filters year range.
- Selects numeric features using training data only:
  - removes constant columns
  - drops high-missing columns (`--missing-threshold`)
  - drops highly correlated columns (`--correlation-threshold`)
- Builds complete year grid per country and imputes numeric gaps with interpolation + forward/backward fill.

**Run examples**
```bash
# process all folds
python scripts/1_time_series_data_preprocessing.py

# process only one fold
python scripts/1_time_series_data_preprocessing.py --fold 0
```

Optional arguments:
- `--owid-dir`, `--policy-input`, `--output-dir`
- `--start-year`, `--end-year`
- `--missing-threshold`, `--correlation-threshold`
- `--fold`

---

## 4) `scripts/2_data_preparation_and_time_series_scaling.py`
**Purpose**
- Builds model-ready train/test inputs from policy metadata + preprocessed time-series files.
- Uses train-fold countries only to fit time-series scaling, then applies the scaler to train/test countries.
- Generates ablation datasets for multiple history windows.

**Main inputs (default)**
- `data/csv/group_kfold_assignments.csv`
- `data/time_series/kfold/fold_{i}/train/*.csv`
- `data/time_series/kfold/fold_{i}/test/*.csv`

**Main outputs**
- `data/model_input/kfold/fold_{i}/scaled_train_time_series.csv`
- `data/model_input/kfold/fold_{i}/scaled_test_time_series.csv`
- `data/model_input/kfold/fold_{i}/scaler.pkl`
- `data/model_input/kfold/fold_{i}/window_{w}/train.jsonl`
- `data/model_input/kfold/fold_{i}/window_{w}/test.json`
- `data/model_input/kfold/fold_{i}/manifest.json`
- `data/model_input/kfold/preparation_summary.json`

**Ablation windows**
- Default windows are `1,2,5,10` and can be changed via `--windows`.
- Each window directory contains aligned train/test data for that specific history length.

**Run examples**
```bash
# process all folds with default windows (1,2,5,10)
python scripts/2_data_preparation_and_time_series_scaling.py

# process one fold with explicit ablation windows
python scripts/2_data_preparation_and_time_series_scaling.py --fold 0 --windows 1,2,5,10
```

Optional arguments:
- `--policy-input`, `--time-series-dir`, `--output-dir`
- `--windows`
- `--negative-samples`
- `--fold`
- `--seed`

---

## Suggested order
1. Run `0_split_test_set.py`
2. Run `0_visualize_data_distribution.py`
3. Run `1_time_series_data_preprocessing.py`
4. Run `2_data_preparation_and_time_series_scaling.py`
5. Run `3_build_chunk_vectordb.py`
6. Run `4_train_siamese.py`
7. Run `4_summary_only_baselines.py` (summary-only baselines)
8. Run `4_generate_role_agents_summary.py` (role-agent summary retrieval evaluation)
9. Run `4_seekpolicy.py` (RAG summary retrieval evaluation)

---

## 5) `scripts/4_train_siamese.py`
**Purpose**
- Runs retrieval training/evaluation experiments from prepared per-fold model inputs.
- Supports experiment matrix over:
  - backbone: `climatebert/distilroberta-base-climate-f`, `sentence-transformers/all-distilroberta-v1`
  - loss: `triplet`, `contrastive`
  - window: `1,2,5,10`
- Logs train/validation loss convergence and evaluates retrieval quality per fold/window.
- Uses same-fold prepared files (`train.jsonl` + `test.json` from the same `fold_i/window_w` directory).
- Uses chunk corpus from persisted vector DB filtered by the current fold metadata.

**Main inputs (default)**
- `data/csv/group_kfold_assignments.csv`
- `data/model_input/kfold/fold_{i}/window_{w}/train.jsonl`
- `data/model_input/kfold/fold_{i}/window_{w}/test.json`
- `data/vectorstore/policy_chunks_chroma/` (persisted chunk DB from `scripts/3_build_chunk_vectordb.py`)

**Main outputs**
- Per-run logs/checkpoints/metrics:
  - `results/retrieval_experiments/fold_{i}/window_{w}/<backbone>/<loss>/`
- Aggregated outputs:
  - `results/retrieval_experiments/all_fold_results.csv`
  - `results/retrieval_experiments/summary_mean_metrics.csv`
  - `results/retrieval_experiments/paired_significance_tests.csv`
  - `results/retrieval_experiments/run_metadata.json`

**Metrics**
- `Hit@k`, `Precision@k`, `NDCG@k` (default k: `1,5,10`).
- Includes paired significance testing across folds:
  - Paired t-test
  - Wilcoxon signed-rank

**Why NDCG/Precision matter for policy retrieval**
- `Hit@k` only checks if the right policy appears somewhere in top-k.
- `NDCG@k` rewards putting the truly relevant/actionable policy near rank 1.
- `Precision@k` penalizes noisy top-k lists where only one item is relevant.

**Run examples**
```bash
# full training matrix (all folds, default windows/backbones/losses)
python scripts/4_train_siamese.py --max-epochs 10

# single fold, single window
python scripts/4_train_siamese.py --fold 0 --windows 1 --max-epochs 10

# custom backbone/loss/window set
python scripts/4_train_siamese.py \
  --windows 1,5 \
  --backbones climatebert/distilroberta-base-climate-f,sentence-transformers/all-distilroberta-v1 \
  --losses triplet,contrastive \
  --max-epochs 12

# hyperparameter tuning before final training
python scripts/4_train_siamese.py \
  --fold 0 \
  --windows 1 \
  --backbones climatebert/distilroberta-base-climate-f \
  --losses triplet,contrastive \
  --tune-hyperparams \
  --tune-max-trials 12 \
  --tune-max-epochs 3 \
  --tune-patience 2

# tune once (single fold/window, fixed backbone), save shared tuned params by loss
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

# run all folds/windows using shared tuned params from the previous command
python scripts/4_train_siamese.py \
  --backbones climatebert/distilroberta-base-climate-f \
  --losses triplet,contrastive \
  --use-tuned-hparams \
  --shared-tuned-hparams \
  --max-epochs 10

# eval-only mode (uses existing checkpoint auto-discovered under output-dir)
python scripts/4_train_siamese.py \
  --fold 0 \
  --windows 1 \
  --backbones climatebert/distilroberta-base-climate-f \
  --losses triplet \
  --eval-only

# eval-only mode with explicit checkpoint
python scripts/4_train_siamese.py \
  --fold 0 \
  --windows 1 \
  --backbones climatebert/distilroberta-base-climate-f \
  --losses triplet \
  --eval-only \
  --eval-checkpoint results/retrieval_experiments/fold_0/window_1/climatebert_distilroberta-base-climate-f/triplet/checkpoints/final/trial_001/best.ckpt

# resolve collection from chunk DB manifest
python scripts/4_train_siamese.py --chunk-vectordb-collection auto
```

Optional arguments:
- `--policy-input`, `--model-input-dir`, `--output-dir`
- `--fold`, `--windows`, `--backbones`, `--losses`, `--k-values`
- `--chunk-vectordb-dir`, `--chunk-vectordb-collection`
- `--batch-size`, `--val-split`, `--max-epochs`, `--patience`
- `--lr`, `--weight-decay`, `--time-series-hidden-size`, `--sector-embedding-dim`, `--embedding-dim`, `--dropout`
- `--margin` (triplet), `--temperature` (contrastive)
- tuning: `--tune-hyperparams`, `--tune-max-epochs`, `--tune-patience`, `--tune-max-trials`,
  `--tune-lr`, `--tune-weight-decay`, `--tune-time-series-hidden-size`, `--tune-sector-embedding-dim`,
  `--tune-embedding-dim`, `--tune-dropout`, `--tune-margin`, `--tune-temperature`,
  `--shared-tuned-hparams`, `--shared-tuned-hparams-dir`
- eval-only: `--eval-only`, `--eval-checkpoint`
- reproducibility/runtime: `--seed`, `--deterministic`, `--non-deterministic`, `--num-workers`, `--device`
- `--baseline-backbone`, `--baseline-loss`, `--baseline-window` (for paired statistical comparison)

**Notes**
- In eval-only mode, `--tune-hyperparams` is not allowed.
- `--shared-tuned-hparams` with `--tune-hyperparams` requires one source fold (`--fold`) and one source window (`--windows` with a single value).
- With `--use-tuned-hparams --shared-tuned-hparams`, tuned params are loaded from `output-dir/shared_tuned_hparams/<backbone>/<loss>.json` (or from `--shared-tuned-hparams-dir`).
- Eval-only checkpoint resolution order:
  1) `--eval-checkpoint` if provided
  2) `<run_dir>/checkpoints/final/trial_001/best.ckpt`
  3) single match of `<run_dir>/checkpoints/final/trial_*/best.ckpt`

---

## 6) `scripts/4_summary_only_baselines.py`
**Purpose**
- Runs two retrieval baselines that do **not** use time-series features.
- Uses only human-written policy summaries (`Family Summary`) as both query and corpus text.

**Baselines**
- `human_summary_semantic`: SentenceTransformer semantic search (`sentence-transformers/all-distilroberta-v1` by default)
- `climate_policy_radar_bm25`: BM25 keyword retrieval (`rank-bm25`)

**Main inputs (default)**
- `data/csv/group_kfold_assignments.csv`

**Main outputs**
- `results/summary_only_baselines/all_fold_results.csv`
- `results/summary_only_baselines/summary_mean_metrics.csv`
- `results/summary_only_baselines/paired_significance_tests.csv`
- `results/summary_only_baselines/run_metadata.json`

**Metrics**
- Same retrieval metrics as `4_train_siamese.py`:
  - `Hit@k`
  - `Precision@k`
  - `NDCG@k`
- Fold-level paired significance tests against a chosen baseline (`--significance-baseline`):
  - Paired t-test p-value
  - Wilcoxon signed-rank p-value

**Run examples**
```bash
# all folds
python scripts/4_summary_only_baselines.py

# single fold
python scripts/4_summary_only_baselines.py --fold 0

# custom k values
python scripts/4_summary_only_baselines.py --k-values 1,5,10
```

**Dependencies**
```bash
pip install sentence-transformers rank-bm25
```

---

## 7) `scripts/4_generate_role_agents_summary.py`
**Purpose**
- Evaluates role-agent summary retrieval with the same fold-aware metrics pipeline used in other experiment scripts.
- Optionally generates `summarizer_v1` summaries with OpenAI before evaluation.

**Methods compared**
- `role_agent_summary_semantic` (query = role-agent summary column, semantic retrieval)
- `human_summary_semantic` (query = human summary column, semantic retrieval baseline)
- `climate_policy_radar_bm25` (query = human summary column, BM25 baseline)

**Main inputs (default)**
- `data/csv/group_kfold_assignments.csv`

**Main outputs**
- `results/role_agent_summary_experiments/all_fold_results.csv`
- `results/role_agent_summary_experiments/summary_mean_metrics.csv`
- `results/role_agent_summary_experiments/paired_significance_tests.csv`
- `results/role_agent_summary_experiments/run_metadata.json`

**Metrics**
- `Hit@k`, `Precision@k`, `NDCG@k`
- Paired significance tests across folds:
  - paired t-test p-value
  - Wilcoxon signed-rank p-value

**Run examples**
```bash
# evaluate existing role-agent summaries in column summarizer_v1
python scripts/4_generate_role_agents_summary.py --query-column summarizer_v1

# single fold only
python scripts/4_generate_role_agents_summary.py --fold 0 --query-column summarizer_v1

# generate missing role-agent summaries first, then evaluate
python scripts/4_generate_role_agents_summary.py \
  --generate-role-summary \
  --source-column "Family Summary" \
  --query-column summarizer_v1
```

**Environment (`.env`)**
```dotenv
OPENAI_API_KEY=
```

**Dependencies**
```bash
pip install sentence-transformers rank-bm25 openai python-dotenv
```

---

## 8) `scripts/4_seekpolicy.py`
**Purpose**
- Replaces the legacy RAG summary script with a production CLI.
- Supports optional persisted Chroma + OpenAI RAG summary generation and fold-aware retrieval evaluation.

**Methods compared**
- `rag_summary_semantic` (query = `RAG_v1_summary`, semantic retrieval)
- `human_summary_semantic` (human summary semantic baseline)
- `climate_policy_radar_bm25` (BM25 baseline)

**Main inputs (default)**
- `data/csv/group_kfold_assignments.csv`

**Main outputs**
- `results/seekpolicy_experiments/all_fold_results.csv`
- `results/seekpolicy_experiments/summary_mean_metrics.csv`
- `results/seekpolicy_experiments/paired_significance_tests.csv`
- `results/seekpolicy_experiments/run_metadata.json`

**Metrics**
- `Hit@k`, `Precision@k`, `NDCG@k`
- Paired fold-level significance tests:
  - paired t-test p-value
  - Wilcoxon signed-rank p-value

**Run examples**
```bash
# evaluate existing RAG summaries
python scripts/4_seekpolicy.py --query-column RAG_v1_summary

# evaluate one fold
python scripts/4_seekpolicy.py --fold 0 --query-column RAG_v1_summary

# generate missing RAG summaries then evaluate
python scripts/4_seekpolicy.py \
  --generate-rag-summary \
  --source-column "Family Summary" \
  --query-column RAG_v1_summary
```

**Environment (`.env`)**
```dotenv
OPENAI_API_KEY=
```

**Dependencies**
```bash
pip install sentence-transformers rank-bm25 openai python-dotenv langchain-openai langchain-community langchain-chroma chromadb
```
