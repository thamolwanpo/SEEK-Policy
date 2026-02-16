# Test Instructions

## Scope
This folder contains verification tests for dataset splitting behavior.

Current test coverage:
- `test_split_test_set.py`: validates leakage-safe family grouping and grouped K-fold train/test consistency.
- `test_time_series_data_preprocessing.py`: validates leakage-safe train-only feature selection in fold preprocessing.
- `test_data_preparation_and_time_series_scaling.py`: validates fold-aware model-input generation and ablation windows (`1,2,5,10`).

Documentation updated (2026-02-14):
- Added/updated logging in `changes.md` and file-level usage notes in `instructions.md`.

## Prerequisites
- Python 3.8+
- Project dependencies installed (`pandas`, `numpy`, `scikit-learn`, `joblib`)

## Run all tests in this folder
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Run only split test verification
```bash
python tests/test_split_test_set.py -v
```

## Run only time-series preprocessing leakage verification
```bash
python tests/test_time_series_data_preprocessing.py -v
```

## Run only model-input preparation + ablation verification
```bash
python tests/test_data_preparation_and_time_series_scaling.py -v
```

## What the split test verifies (reviewer-focused)
1. K-fold output assigns each family to exactly one fold.
2. Per-fold train/test files are generated for each fold.
3. No family leakage between train and test inside every fold (`Family ID` overlap must be 0).

## What the time-series preprocessing test verifies
1. Feature selection is fit on train-fold countries only.
2. A feature present only in test-country rows is dropped from fold outputs.
3. Fold output files are generated in train/test directories.

## What the model-input preparation test verifies
1. Fold-aware script execution succeeds end-to-end from synthetic grouped assignments + fold time-series files.
2. Ablation outputs are generated for windows `1,2,5,10`.
3. Each generated sample has a time axis length equal to the selected window.
