# PS4 environment and data

From the repository root, create the shared Conda environment with `conda env create -f environment.yml`. If you already have the `ps3` environment, update it with `conda env update -n ps3 -f environment.yml`. PyArrow supports the Parquet cache. Exercises 4 and 5 use DALEX 1.8.0 for partial dependence and Shapley explanations.

```bash
conda activate ps3
python -m pip install --no-build-isolation -e .
python -c 'from importlib.metadata import version; print("DALEX", version("dalex")); print("PyArrow", version("pyarrow"))'
python -m pytest
pre-commit run --all-files
```

## Prepare the shared claims data

The first command downloads the instructor's OpenML data, applies the repository's transformations, and saves a local cache. Reuse the cache with the second command if it already exists.

```bash
python -m analyses.ps3_script check-data --save-data data/ps4_claims.parquet
python -m analyses.ps3_script check-data --data data/ps4_claims.parquet
```

Expect 678,013 policies (542,286 training and 135,727 test). Use the fixed `IDpol` split, `ClaimAmountCut / Exposure` as the pure premium target, `Exposure` as the weight, and Tweedie power 1.5. Keep data caches, fitted models, and per-policy predictions local.

## Reproduce Exercise 1 for the other exercises

```bash
python -m analyses.ps3_script ps4-ex1 --data data/ps4_claims.parquet --results-dir data/ps4_ex1
```

Use the selected parameters and transformed feature order in `data/ps4_ex1/ps4_ex1_monotonicity_metrics.json`. Join `data/ps4_ex1/ps4_ex1_test_predictions.csv` by `IDpol` for paired evaluation. For model explanations, use the fitted preprocessing for each model and use the same background sample when comparing models.
