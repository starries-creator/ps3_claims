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

For Exercise 4, obtain both fitted pipelines in the same Python process.

```python
import pandas as pd

from analyses.lgbm_monotonicity import run_monotonicity_comparison

claims = pd.read_parquet("data/ps4_claims.parquet")
models, _, _, _ = run_monotonicity_comparison(claims, "data/ps4_ex1")
constrained_lgbm = models["constrained_lgbm"]
unconstrained_lgbm = models["unconstrained_lgbm"]
```

Pass the same raw feature columns to both pipelines. Each pipeline applies its fitted preprocessing before predicting.

## PS4 pull request checks

Put each exercise's implementation in a descriptive Python module, such as
`analyses/learning_curve.py`. Add a labelled `# %%` section and the necessary
imports and calls to `analyses/ps3_script.py` for a reproducible run. Follow the
instructor's named file or format when specified. For example, the metrics
function belongs in `ps3/evaluation/_evaluate_predictions.py`.

Use a title that says what changed, such as **PS4 Ex2 — Plot the LightGBM
learning curve**. Use descriptive names for new files and public functions.
Keep internal teammate labels out of names and PR descriptions.

For the four teammates who work on a single exercise, the
[acceptance rubric](.github/ps4/rubric.json) selects that exercise by GitHub PR
author from PR #16 onward. The maintainer and any other author can use `PS4 ExN`
in the PR title, description or branch name. The rubric also recognizes exercise
paths and lists the handout, individual questions and earlier Ex1 review. It
records `met`, `missing` or `needs_review` with evidence in the Actions summary.

The executable job runs the repository's tests on the proposed merge. For Ex1
it also trains both LightGBM variants on a small, deterministic synthetic dataset
and checks weighting, feature order, monotonicity, training-only CV and paired
test predictions. For later exercises, include exercise-specific tests under
`tests/` with the implementation so that the executable job runs them.

After a PS4 PR is merged, the repository owner can comment `/recheck merged-pr`
on that PR. This runs the general reviewer against the original text diff and
the assignment checks against the recorded merge commit. Results are written to
the two Actions summaries. The general review also lists archived findings in the
job log. The merge decision remains with the maintainer.
