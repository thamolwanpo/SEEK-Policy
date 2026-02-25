# Experiments: Siamese Retrieval Network

This file describes the experiment protocol for `scripts/4_train_siamese.py`.

## Model framing

The training script implements a **siamese-style dual-encoder retrieval network**:
- Text tower: transformer backbone (`climatebert/...` or `all-distilroberta-v1`)
- Time-series tower: MLP projection over flattened windowed features + sector embedding
- Shared retrieval space: both towers project into a common embedding space
- Training objectives: `triplet` or `contrastive`

Although this is cross-modal (text vs time-series), it follows siamese retrieval training behavior by learning aligned embeddings and distance-based ranking.

## Input prerequisites

Run these scripts first:
1. `python scripts/0_split_test_set.py`
2. `python scripts/1_time_series_data_preprocessing.py`
3. `python scripts/2_data_preparation_and_time_series_scaling.py`

Expected training/eval inputs:
- `data/csv/group_kfold_assignments.csv`
- `data/model_input/kfold/fold_<i>/window_<w>/train.jsonl`
- `data/model_input/kfold/fold_<i>/window_<w>/test.json`

## Experiment matrix

Default matrix is:
- Folds: all discovered folds
- Windows: `1,2,5,10`
- Backbones:
  - `climatebert/distilroberta-base-climate-f`
  - `sentence-transformers/all-distilroberta-v1`
- Losses:
  - `triplet`
  - `contrastive`

Total combinations per full run: `n_folds × 4 windows × 2 backbones × 2 losses`.

## Reproducibility setup

The script sets deterministic/reproducible behavior by:
- fixed seed (`--seed`, default `42`)
- `PYTHONHASHSEED`
- NumPy, Python `random`, Torch CPU/GPU seeds
- Lightning `seed_everything(..., workers=True)`
- deterministic CuDNN setup and deterministic algorithm mode (default on)
- deterministic DataLoader worker seeding (`worker_init_fn`)

Use `--non-deterministic` only when prioritizing speed over strict reproducibility.

## Training and evaluation metrics

Per fold/combination, the script logs:
- training convergence: `train_loss`, `val_loss` (Lightning CSV logs)
- retrieval metrics: `Hit@k`, `Precision@k`, `NDCG@k`, `MRR@k`

Retrieval corpus is chunk-based:
- Policy texts are loaded from `text_file_path`
- Text is split into chunks (`--chunk-size`, `--chunk-overlap`)
- Query = time-series embedding, corpus = chunk embeddings from text tower
- Chunk scores are aggregated to document score by max, then ranked by `Document ID`

Why ranking metrics matter:
- `Hit@k` can be too permissive for policy use-cases.
- `NDCG@k` rewards placing truly relevant/actionable policies near the top.
- `Precision@k` measures concentration of relevant results in top ranks.

## Statistical significance across folds

After all fold runs complete, script outputs paired significance tests against a configurable baseline:
- Paired t-test p-values
- Wilcoxon signed-rank p-values

Configured by:
- `--baseline-backbone`
- `--baseline-loss`
- `--baseline-window`

## Output artifacts

Per-run directories:
- `results/retrieval_experiments/fold_<i>/window_<w>/<backbone>/<loss>/`
  - `tuning_results.json` (when `--tune-hyperparams` is enabled)

Shared tuned hyperparameters (optional):
- `results/retrieval_experiments/shared_tuned_hparams/<backbone>/<loss>.json`
  - generated when `--shared-tuned-hparams` is used during tuning
  - consumed by all folds/windows when running with `--use-tuned-hparams --shared-tuned-hparams`

Aggregated files:
- `results/retrieval_experiments/all_fold_results.csv`
- `results/retrieval_experiments/summary_mean_metrics.csv`
- `results/retrieval_experiments/paired_significance_tests.csv`
- `results/retrieval_experiments/run_metadata.json`

## Example commands

Full default sweep:
```bash
python scripts/4_train_siamese.py --max-epochs 10
```

With hyperparameter tuning then final training:
```bash
python scripts/4_train_siamese.py \
  --tune-hyperparams \
  --tune-max-trials 24 \
  --tune-max-epochs 5 \
  --max-epochs 10
```

Tune once (single fold/window, fixed backbone) and reuse everywhere:
```bash
# pass 1: tune once and save shared params by loss
python scripts/4_train_siamese.py \
  --fold 0 \
  --windows 5 \
  --backbones climatebert/distilroberta-base-climate-f \
  --losses triplet,contrastive \
  --tune-hyperparams \
  --tune-only \
  --shared-tuned-hparams \
  --tune-max-trials 12 \
  --tune-max-epochs 3

# pass 2: train/evaluate all folds/windows with shared tuned params
python scripts/4_train_siamese.py \
  --backbones climatebert/distilroberta-base-climate-f \
  --losses triplet,contrastive \
  --use-tuned-hparams \
  --shared-tuned-hparams \
  --max-epochs 10
```

Single-fold smoke run:
```bash
python scripts/4_train_siamese.py --fold 0 --windows 1 --backbones climatebert/distilroberta-base-climate-f --losses triplet --max-epochs 5
```

Custom matrix:
```bash
python scripts/4_train_siamese.py \
  --windows 1,2,5,10 \
  --backbones climatebert/distilroberta-base-climate-f,sentence-transformers/all-distilroberta-v1 \
  --losses triplet,contrastive \
  --k-values 1,5,10 \
  --max-epochs 12 \
  --seed 42
```

Fold protocol:
- For held-out fold `f`, training uses all folds `!= f` and testing uses fold `f`.
- Example with folds `1..5`: train on `1..4`, test on `5`.

---

# Experiments: Role-Agent Summary Retrieval

This section describes the protocol for `scripts/4_generate_role_agents_summary.py`.

## Goal

Evaluate chunk-level retrieval quality when the query text is a role-agent summary (for example `role_agent_summary`) and run the experiment by fold and history window.

## Evaluation setup

- Fold split source: `data/csv/group_kfold_assignments.csv`
- Windowed test source: `data/model_input/kfold/fold_<i>/window_<w>/test.json`
- Retrieval corpus for each fold: chunk corpus from persisted Chroma vector DB filtered by the same fold id (`fold == current_fold`)
- Query set for each fold/window: records from `test.json` mapped to policy document ids via `anchor_summary`
- Agent table schema source: `data/model_input/kfold/fold_<i>/scaled_test_time_series.csv` (test-only; can differ by fold)
- Time-series prompt inputs are split into separate `climate_data`, `socio_economics_data`, and `other_data` tables per fold

Compared methods:
- `role_agent_summary_chunk_semantic` (OpenAI embeddings, query from role-agent summary column)

## Metrics

- `Hit@k`
- `Precision@k`
- `NDCG@k`
- `MRR@k`

## Output artifacts

- `results/role_agent_summary_experiments/all_fold_results.csv`
- `results/role_agent_summary_experiments/summary_mean_metrics.csv`
- `results/role_agent_summary_experiments/run_metadata.json`

## OpenAI generation mode (optional)

To generate role-agent summaries from fold/window time-series records before evaluation:
- add `OPENAI_API_KEY` in `.env`
- run with `--generate-role-summary-from-time-series`

## Example commands

Evaluate one fold and one window:
```bash
python scripts/4_generate_role_agents_summary.py \
  --fold 0 \
  --window 1 \
  --chunk-vectordb-collection auto
```

Generate role-agent summaries from time-series chain, then evaluate:
```bash
python scripts/4_generate_role_agents_summary.py \
  --generate-role-summary-from-time-series \
  --save-generated-role-summaries \
  --windows 1,2,5,10 \
  --chunk-vectordb-collection auto
```

---

# Experiments: SEEK-Policy RAG Summary Retrieval

This section documents `scripts/4_seekpolicy.py`.

## Goal

Evaluate retrieval effectiveness when queries are RAG-generated policy summaries (`RAG_v1_summary`) and compare against semantic/BM25 baselines.

## Pipeline modes

- **Evaluation-only**: uses an existing `RAG_v1_summary` column.
- **Generate + evaluate**: uses persisted Chroma retrieval and OpenAI summarization to generate missing RAG summaries first.

## Fold protocol

- Corpus per fold: rows with `fold != current_fold` (human summaries)
- Queries per fold: rows with `fold == current_fold`

Compared methods:
- `rag_summary_semantic`
- `human_summary_semantic`
- `climate_policy_radar_bm25`

## Metrics and significance

Metrics:
- `Hit@k`
- `Precision@k`
- `NDCG@k`

Significance:
- paired t-test p-values (fold-aligned)
- Wilcoxon signed-rank p-values (fold-aligned)

## Output artifacts

- `results/seekpolicy_experiments/all_fold_results.csv`
- `results/seekpolicy_experiments/summary_mean_metrics.csv`
- `results/seekpolicy_experiments/paired_significance_tests.csv`
- `results/seekpolicy_experiments/run_metadata.json`

## Environment requirements

Set in `.env`:
- `OPENAI_API_KEY`

## Example commands

Evaluation-only:
```bash
python scripts/4_seekpolicy.py --query-column RAG_v1_summary
```

Generate then evaluate:
```bash
python scripts/4_seekpolicy.py \
  --generate-rag-summary \
  --source-column "Family Summary" \
  --query-column RAG_v1_summary
```
