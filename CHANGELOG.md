# Changes Log

This file tracks notable changes across files in this repository.

## 2026-02-18

### File: scripts/4_seekpolicy.py

- Updated SEEK-Policy generation/evaluation flow to current always-generate behavior:
	- removed optional `--generate-rag-summary` gating; generation now runs before evaluation
	- enabled per-window generation for multi-window runs (`--windows 1,2,5,10`)
	- evaluation now uses the generated summaries associated with each specific window
- Added explicit fold-window completion progress line in this format:
	- `fold, window, left/total`
	- printed after each fold/window completes (including skipped windows)
- Updated generated summary output behavior:
	- single-window run writes `generated_rag_summaries.csv`
	- multi-window run writes `generated_rag_summaries_by_window/window_<w>.csv`

### File: scripts/4_seekpolicy.py

- Refined SEEK-Policy retrieval workflow to the current fold/window protocol and method naming:
	- evaluation method recorded as `rag_summary_chunk_semantic`
	- queries sourced from `data/model_input/kfold/fold_<i>/window_<w>/test.json`
- Added eval metadata filter toggle with safe default OFF:
	- new `--eval-use-metadata-filter` flag enables eval-time metadata filtering only when explicitly requested
	- generation-time progressive metadata fallback remains enabled for RAG summary generation
- Added per-query retrieval trace export:
	- new `--save-retrieval-traces` and `--retrieval-trace-top-k` options
	- trace output under `results/seekpolicy_experiments/fold_<i>/window_<w>/rag_summary_chunk_semantic/retrieval_traces.csv`
- Added fold-level parallel execution and explicit progress logging:
	- new `--fold-parallelism` option
	- query progress logs include `fold/window` and `left/total`
- Added generation table-stage parallelism:
	- new `--table-parallelism` option for parallel table analysis and blending steps

### File: scripts/3_build_chunk_vectordb.py

- Added metadata-only backfill mode for existing persisted Chroma records:
	- new `--update-metadata-only` flow updates `geography` and `sector` by `document_id` without re-embedding
	- new `--metadata-update-batch-size` option for batched metadata updates
- Updated standard chunk metadata payload to include `geography` and `sector` during normal index build.

### File: README.md

- Updated `scripts/4_seekpolicy.py` documentation to match current behavior:
	- removed stale BM25/human-baseline and paired-significance references
	- documented default eval filter OFF behavior
	- added all-fold/all-window run example with trace saving
	- documented trace output paths and current dependency list.
	- updated method name to `rag_summary_semantic`
	- documented always-generate behavior and `fold, window, left/total` progress logs.

### File: instructions.md

- Rewrote operational instructions for `scripts/4_seekpolicy.py`:
	- aligned purpose/inputs/outputs/options with current CLI and protocol
	- added run commands for all folds/windows with trace export (without eval filter)
	- removed stale options no longer present in CLI
	- documented current progress line format: `fold, window, left/total`.

## 2026-02-17

### File: scripts/4_generate_role_agents_summary.py

- Simplified role-agent retrieval to one method only:
	- `role_agent_summary_chunk_semantic`
- Removed non-requested comparison/significance paths from this script:
	- removed human-summary semantic baseline and BM25 baseline
	- removed paired significance test generation/output
- Aligned role-summary generation input formatting with notebook-style table blocks:
	- `[START_OF_TABLE] ... [END_OF_TABLE]`
- Updated generation chain so advisors consume separate domain tables instead of one shared table:
	- `climate_data`
	- `socio_economics_data`
	- `other_data`
- Added fold-specific, test-only table schema behavior for advisor table splitting:
	- domain feature indices are derived from `data/model_input/kfold/fold_<i>/scaled_test_time_series.csv`
	- no train-set schema fallback in role-agent generation
- Renamed ambiguous summary stage variables for clarity:
	- `summary_v0` → `retrieval_summary_draft`
	- `summary_v1` → `retrieval_summary_final`
- Updated default query column name:
	- `summarizer_v1` → `role_agent_summary`

### File: README.md

- Updated `scripts/4_generate_role_agents_summary.py` documentation to match current behavior:
	- single retrieval method only
	- no paired significance output
	- metrics include `MRR@k`
	- generation examples updated to `--generate-role-summary-from-time-series`
	- default query column updated to `role_agent_summary`
	- documented fold-specific, test-only schema source for advisor table splits

### File: experiments.md

- Updated role-agent experiment protocol text:
	- replaced example query name `summarizer_v1` with `role_agent_summary`
	- documented test-only fold schema source (`scaled_test_time_series.csv`)
	- documented per-domain advisor table splitting (climate/socio/other)

### File: instructions.md

- Rewrote operational section for `scripts/4_generate_role_agents_summary.py` to match current script:
	- updated defaults/options/commands/output artifacts
	- removed stale significance and BM25/human-baseline references
	- added fold/window test-only + domain-split behavior notes

## 2026-02-16

### File: README.md

- Added a new top-level `Quick run (end-to-end)` section with:
	- environment setup (`pip install -r requirements.txt`)
	- `.env` note for `OPENAI_API_KEY`
	- ordered pipeline commands from split/preprocess through retrieval experiments.
- Updated `scripts/4_summary_only_baselines.py` documentation to match current script behavior:
	- corrected baseline method names to `human_summary_chunk_semantic` and `chunk_bm25`
	- documented chunk vector DB input requirements
	- added retrieval trace output/options
	- corrected dependencies to OpenAI/LangChain + BM25 stack.

### File: instructions.md

- Added a new `Project quick run (from scratch)` section with a minimal, ordered command sequence for setup and execution.
- Updated `scripts/4_summary_only_baselines.py` operational instructions to reflect current CLI/options and evaluation behavior:
	- OpenAI embedding model option and chunk DB options
	- trace export options
	- corrected default significance baseline name
	- updated dependencies and `OPENAI_API_KEY` requirement.

### File: scripts/4_train_siamese.py

- Updated retrieval corpus fold filtering to use the current fold only (`train_fold_ids = [fold_id]`) so each fold/window run stays within the same prepared fold directory context.
- Added per-query chunk retrieval trace export:
	- new `--save-retrieval-traces` flag to write ranked retrieved chunks per query
	- new `--retrieval-trace-top-k` flag to control how many top chunks are saved per query (`0` resolves to `max(k-values)`)
	- per-run trace output at `results/retrieval_experiments/fold_<i>/window_<w>/<backbone>/<loss>/retrieval_traces.csv`
	- trace rows include fold/window/config, query index/id, target doc id, retrieved rank/doc id/score, relevance flag, and retrieved chunk text.
- Added robust chunk vector DB compatibility for newer/older metadata conventions:
	- collection resolution from `manifest.json` (`--chunk-vectordb-collection auto|manifest`)
	- metadata key fallback support for fold/document identifiers.
- Added eval-only workflow:
	- new `--eval-only` flag to skip training/tuning and run retrieval evaluation only
	- new `--eval-checkpoint` flag for explicit checkpoint path
	- automatic checkpoint discovery under `checkpoints/final/trial_001/best.ckpt` (or a single `trial_*/best.ckpt` candidate).
- Added guardrail: `--eval-only` cannot be combined with `--tune-hyperparams`.
- Updated run metadata fields to reflect same-fold protocol and eval-only execution details.
- Added shared tuned-hyperparameter workflow for tune-once/reuse-many:
	- new `--shared-tuned-hparams` flag to save/load tuned params by `<backbone>/<loss>` across folds/windows
	- new `--shared-tuned-hparams-dir` override for shared tuning file location
	- save path default: `results/retrieval_experiments/shared_tuned_hparams/<backbone>/<loss>.json`
	- loading behavior: with `--use-tuned-hparams --shared-tuned-hparams`, all fold/window runs load from shared files
	- added guardrails so shared tuning generation requires a single source fold and single source window.
- Separated tuning artifacts from fold training outputs:
	- tuning checkpoints/logs and `tuning_results.json` now save under `results/retrieval_experiments/tuning_runs/fold_<i>/window_<w>/<backbone>/<loss>/`
	- fold run directories under `results/retrieval_experiments/fold_<i>/...` are now reserved for final training/evaluation artifacts
	- non-shared `--use-tuned-hparams` now loads per-run tuned params from the dedicated `tuning_runs` path.

### File: README.md

- Expanded `scripts/4_train_siamese.py` documentation with complete run settings:
	- standard training examples
	- tuning examples
	- tune-once/reuse-many shared hyperparameter example
	- eval-only examples (auto and explicit checkpoint)
	- chunk vector DB options and collection auto-resolution
	- added notes for eval-only checkpoint lookup order and constraints
	- documented shared tuned-hyperparameter options and constraints.

### File: instructions.md

- Updated operational instructions for `scripts/4_train_siamese.py` to document:
	- same-fold train/test usage from `data/model_input/kfold/fold_<i>/window_<w>/`
	- all relevant CLI option groups (core, chunk DB, tuning, eval-only, reproducibility)
	- eval-only checkpoint lookup behavior and constraints
	- shared tuned-hyperparameter workflow and CLI constraints.

## 2026-02-14

### File: scripts/0_split_test_set.py

- Replaced the original random 100-sample split logic with a leakage-safe split strategy.
- Added family-based grouping (`Family ID` with fallbacks) so all versions of the same law stay in the same partition.
- Added strict temporal holdout selection using newest year buckets until the test target is reached (default: 1,000).
- Added grouped K-fold assignment output for robust evaluation (`data/csv/group_kfold_assignments.csv`).
- Added deterministic CLI options (`--target-test-size`, `--n-folds`, `--seed`, input/output paths).
- Added validation checks for group overlap leakage and empty split protection.

### File: scripts/0_visualize_data_distribution.py

- Refactored notebook-style script into a readable, function-based Python script.
- Updated data paths to align with the new split pipeline (`data/csv/all_data_en.csv`, `data/csv/data.csv`, `data/csv/test.csv`, `data/csv/region.csv`).
- Kept region/country/sector distribution outputs while saving figures to `figures/`.
- Added graceful fallback when `geopandas` is unavailable (country-map plotting is skipped, other plots still run).

### File: scripts/0_split_test_set.py

- Updated split behavior to grouped K-fold only (removed single global `data.csv`/`test.csv` generation).
- Added per-fold train/test outputs under `data/csv/kfold/` as `fold_<i>_train.csv` and `fold_<i>_test.csv`.
- Kept family-group leakage protection across every fold and added guard for invalid `n_folds > unique groups`.

### File: scripts/0_visualize_data_distribution.py

- Added `--fold` option to visualize a specific K-fold split from `data/csv/kfold`.
- Added fallback to legacy `data.csv`/`test.csv` only when fold files are not present.

### File: tests/test_split_test_set.py

- Updated test coverage to validate grouped K-fold per-fold train/test outputs and no family leakage in each fold.
- Added an explicit reviewer-case assertion that a multi-version family (e.g., 2008 and 2018 versions) is assigned to exactly one fold.

### File: instructions.md

- Fixed split script example command typo (`--n-folds 5`).
- Clarified leakage guarantee with an explicit example: if 2008 and 2018 versions belong to the same law family, they cannot be split between train and test.

### File: tests/TEST_INSTRUCTIONS.md

- Updated test documentation to match grouped K-fold behavior and per-fold leakage checks.
- Kept test run commands aligned with working invocation in this environment.

### File: scripts/0_visualize_data_distribution.py

- Removed legacy fallback to `data.csv`/`test.csv`; visualization now requires fold-specific files in `data/csv/kfold/`.
- Added clear error messaging when requested fold files are missing.
- Updated behavior to loop through all available folds by default and save outputs to `figures/kfold/fold_<i>/`.
- Added optional `--fold` for single-fold visualization.
- Added compatibility handling for geopandas versions where `naturalearth_lowres` dataset helper is unavailable (map plot is skipped instead of failing).
- Removed benchmark-set usage from fold visualizations; comparisons are now train vs test only.
- Refactored outputs to consolidated all-fold summary figures under `figures/kfold_summary/` (one figure per graph type).

### File: scripts/1_time_series_data_preprocessing.py

- Replaced notebook-exported Colab code with a production CLI script.
- Added grouped K-fold-aware preprocessing driven by `data/csv/group_kfold_assignments.csv`.
- Added leakage protection by fitting feature filtering (missingness, constant, correlation) on train countries only for each fold.
- Added fold-specific outputs under `data/time_series/kfold/fold_<i>/{train,test}/` and per-fold manifest JSON files.
- Added preprocessing summary output at `data/time_series/kfold/preprocessing_summary.json`.

### File: tests/test_time_series_data_preprocessing.py

- Added regression test to verify train-only feature selection does not retain a feature observed only in test countries.
- Added end-to-end script invocation check for fold-specific output file generation.

### File: scripts/2_data_preparation_and_time_series_scaling.py

- Replaced notebook-exported Colab script with a CLI pipeline aligned to grouped k-fold outputs.
- Updated inputs to use `data/csv/group_kfold_assignments.csv` and fold-specific time-series files from `data/time_series/kfold/fold_<i>/{train,test}`.
- Added leakage-safe scaling: `StandardScaler` is fit on train-fold country rows only, then applied to both train/test rows.
- Added multi-window ablation generation for configurable history windows (default: `1,2,5,10`).
- Added fold/window outputs under `data/model_input/kfold/fold_<i>/window_<w>/` with `train.jsonl` and `test.json`.
- Added fold manifest and global preparation summary files.

### File: tests/test_data_preparation_and_time_series_scaling.py

- Added regression test that runs the script end-to-end on synthetic fold data.
- Verifies outputs are generated for windows `1,2,5,10`.
- Verifies each output sample has the expected time dimension equal to the selected window.

### File: scripts/4_train_siamese.py

- Replaced notebook-exported Colab script with a production CLI experiment runner.
- Added fold-aware training/evaluation over prepared inputs from `scripts/2_data_preparation_and_time_series_scaling.py`.
- Added support for multi-window ablations (`1,2,5,10`) and flexible window selection via `--windows`.
- Added backbone sweep support including:
	- `climatebert/distilroberta-base-climate-f`
	- `sentence-transformers/all-distilroberta-v1`
- Added loss sweep support (`triplet`, `contrastive`) for retrieval-task alignment.
- Added end-to-end experiment matrix execution across folds/windows/backbones/losses.
- Added train/validation convergence logging (`train_loss`, `val_loss`) with CSV logs and best-checkpoint saving per run.
- Added retrieval evaluation metrics per fold:
	- `Hit@k`
	- `Precision@k`
	- `NDCG@k`
- Added aggregated summary report across folds (`summary_mean_metrics.csv`).
- Added paired statistical comparison across folds versus a configurable baseline:
	- Paired t-test p-values
	- Wilcoxon signed-rank p-values
	- Output in `paired_significance_tests.csv`

### File: README.md

- Added documentation for `scripts/4_train_siamese.py` including experiment matrix, metrics, outputs, and run examples.
- Added rationale for NDCG/Precision in policy retrieval quality assessment.

### File: instructions.md

- Added operational instructions for `scripts/4_train_siamese.py`.
- Documented new CLI options for folds/windows/backbones/losses and baseline-based significance testing.

### File: scripts/4_train_siamese.py

- Renamed training script from `scripts/3_train_mlp.py` to `scripts/4_train_siamese.py`.
- Added stronger reproducibility controls:
	- global Python/NumPy/Torch seeding
	- Lightning worker seeding
	- deterministic DataLoader worker init
	- deterministic CUDA/CuDNN configuration (with opt-out via `--non-deterministic`)
- Added metadata fields to persist reproducibility settings (`seed`, `deterministic`, `model_type`).

### File: scripts/4_summary_only_baselines.py

- Added a new summary-only retrieval baseline runner that does not use time-series inputs.
- Implemented two baselines:
	- `human_summary_semantic` using `SentenceTransformer` semantic search
	- `climate_policy_radar_bm25` using `BM25Okapi` keyword retrieval
- Added fold-aware evaluation using `fold` assignments from `data/csv/group_kfold_assignments.csv`.
- Reused the same ranking metrics as the siamese script:
	- `Hit@k`
	- `Precision@k`
	- `NDCG@k`
- Added output artifacts under `results/summary_only_baselines/`:
	- `all_fold_results.csv`
	- `summary_mean_metrics.csv`
	- `run_metadata.json`
- Updated default SentenceTransformer encoder to `sentence-transformers/all-distilroberta-v1`.
- Added fold-level paired significance testing across baselines:
	- paired t-test p-values
	- Wilcoxon signed-rank p-values
- Added significance output file:
	- `paired_significance_tests.csv`

### File: README.md

- Added documentation for `scripts/4_summary_only_baselines.py`, including purpose, baselines, metrics, outputs, dependencies, and run commands.
- Added the new script to the suggested pipeline order.

### File: instructions.md

- Added operational instructions for `scripts/4_summary_only_baselines.py` with inputs/outputs/options and dependency requirements.

### File: scripts/4_generate_role_agents_summary.py

- Replaced the notebook-exported Colab script with a production CLI workflow.
- Added fold-aware retrieval evaluation using the same metric set as other experiment scripts:
	- `Hit@k`
	- `Precision@k`
	- `NDCG@k`
- Added paired statistical significance testing across folds:
	- paired t-test p-values
	- Wilcoxon signed-rank p-values
- Added method-level comparison outputs for:
	- `role_agent_summary_semantic`
	- `human_summary_semantic`
	- `climate_policy_radar_bm25`
- Added optional OpenAI-based role-summary generation mode (`--generate-role-summary`) with `.env` key loading.
- Added output artifacts under `results/role_agent_summary_experiments/`:
	- `all_fold_results.csv`
	- `summary_mean_metrics.csv`
	- `paired_significance_tests.csv`
	- `run_metadata.json`

### File: .env

- Added `.env` template file with `OPENAI_API_KEY=` for local OpenAI credential configuration.

### File: README.md

- Added documentation for `scripts/4_generate_role_agents_summary.py`, including metrics, significance testing, run commands, and optional OpenAI generation usage.
- Added the script to the documented script list and suggested execution order.

### File: instructions.md

- Added full operational instructions for `scripts/4_generate_role_agents_summary.py`, including required columns, CLI options, outputs, and environment setup.

### File: experiments.md

- Added a dedicated experiment protocol section for role-agent summary retrieval experiments, including methods, fold setup, metrics, significance testing, and output artifacts.

### File: scripts/4_seekpolicy.py

- Replaced legacy `scripts/3_Generate_RAG_Summary.py` with a production CLI script named `scripts/4_seekpolicy.py`.
- Added optional persisted Chroma + OpenAI RAG summary generation mode (`--generate-rag-summary`) for `RAG_v1_summary`.
- Added fold-aware retrieval evaluation using the same metric set as other retrieval scripts:
	- `Hit@k`
	- `Precision@k`
	- `NDCG@k`
- Added paired significance tests across folds:
	- paired t-test p-values
	- Wilcoxon signed-rank p-values
- Added output artifacts under `results/seekpolicy_experiments/`:
	- `all_fold_results.csv`
	- `summary_mean_metrics.csv`
	- `paired_significance_tests.csv`
	- `run_metadata.json`

### File: scripts/3_Generate_RAG_Summary.py

- Removed the legacy notebook-exported script after migration to `scripts/4_seekpolicy.py`.

### File: README.md

- Added documentation for `scripts/4_seekpolicy.py` including methods, outputs, metrics, significance testing, run commands, and required `.env` variables.
- Added `4_seekpolicy.py` to script list and suggested execution order.

### File: instructions.md

- Added full operational instructions for `scripts/4_seekpolicy.py` including generation/evaluation modes, required columns, options, outputs, and environment setup.

### File: experiments.md

- Added a dedicated experiment protocol section for SEEK-Policy RAG summary retrieval using `scripts/4_seekpolicy.py`.

### File: experiments.md

- Added a dedicated experiment protocol document for the siamese-style retrieval network.
- Documented full fold/window/backbone/loss matrix, metrics (`Hit@k`, `Precision@k`, `NDCG@k`), significance testing, and output artifacts.

---

## Template for future entries

### File: <path/to/file>

- What changed
- Why it changed
- Output/impact (if any)
