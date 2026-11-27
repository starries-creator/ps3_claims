# PS3 — Feature engineering and claims modelling

This submission covers the fixed train/test split, the Winsorizer transformer and the motor claims model comparison. The [official problem set](https://languid-citipati-d0d.notion.site/PS-3-Feature-engineering-modelling-2ac32dea82c1814dad25e6d3c2af3b4e) gives the three exercise prompts.

## Install and run

```bash
git clone https://github.com/starries-creator/ps3_claims.git
cd ps3_claims
conda env create -f environment.yml
conda activate ps3
python -m pip install --no-build-isolation -e .
python -m pytest
python -m analyses.ps3_script all --save-data data/shared_final.parquet --results-dir data/reproduced
```

The last command loads the instructor's OpenML data, fits the four models and writes the comparison table and Lorenz figure to `data/reproduced/`. Once the cache exists, rerun using it as input.

```bash
python -m analyses.ps3_script all --data data/shared_final.parquet --results-dir data/reproduced
```

## Exercise 1 — Fixed train/test split

`create_sample_split` labels rows as `train` or `test` by hashing one or more normalized ID columns. A policy keeps its assignment when rows are shuffled or new policies are added. For this dataset, splitting on `IDpol` with a target training fraction of 80% gives **542,286 training policies** and **135,727 test policies**. [Implementation](ps3/data/_sample_split.py) · [Split tests](tests/data/test_sample_split.py) · [Exercise 1 PR](https://github.com/starries-creator/ps3_claims/pull/10)

## Exercise 2 — Winsorizer

`Winsorizer.fit` learns the lower and upper quantile cutoffs from the training data. `transform` clips values to those same cutoffs when applied to new data. In the unit test, fitting the 25th and 75th percentiles on `[0, 10, 20, 30, 100]` gives cutoffs of 10 and 30. The transformer then clips `[-100, 50, 1000]` to `[10, 30, 30]`. [Implementation](ps3/preprocessing/_winsorizer.py) · [Transformer tests](tests/preprocessing/test_winsorizer.py) · [Exercise 2 PR](https://github.com/starries-creator/ps3_claims/pull/8)

## Exercise 3 — Improve the model

All four models use the instructor's data transformation and the same fixed split from Exercise 1. The table reports mean Exposure-weighted Tweedie deviance at power 1.5. Lower is better. [Combined results](results/metrics.csv) · [Data transformation](ps3/data/_load_transform.py) · [Evaluation code](analyses/claims_evaluation.py) · [Integrated runner](analyses/ps3_script.py)

| Model | Train deviance | Test deviance |
| --- | ---: | ---: |
| [Tweedie GLM benchmark](results/evaluated/tweedie_glm_metrics.json) | 73.4855 | 74.8970 |
| [Quadratic Tweedie GLM](results/evaluated/parametric_glm_metrics.json) | 73.4660 | 74.8145 |
| [Initial LightGBM](results/evaluated/lightgbm_initial_metrics.json) | 64.5773 | 74.7513 |
| [Tuned LightGBM](results/evaluated/lightgbm_tuned_metrics.json) | 66.4130 | **74.2916** |

### Why divide by Exposure?

`PurePremium = ClaimAmountCut / Exposure` expresses clipped claims per unit of coverage, so policies with different coverage durations are comparable. Fitting and evaluation weight each policy by `Exposure`. To get the predicted portfolio claim total, multiply each policy's predicted pure premium by its exposure and sum the amounts. [Shared evaluation](analyses/claims_evaluation.py) · [Comparison report](results/final_comparison.md)

### What changed in the GLM?

The benchmark uses a linear term for `BonusMalus` and treats `VehPower` as categorical. The submitted GLM uses linear and squared terms for both variables. Train and test deviance fall slightly, with test deviance moving from 74.8970 to 74.8145. [Benchmark coefficients](results/tweedie_glm_coefficients.csv) · [Quadratic implementation](analyses/parametric_glm.py) · [GLM notes and variant results](results/parametric_glm_notes.md)

### How did we check for overfitting?

The GLM analysis compares train and test deviance, checks cross-validation results and inspects fitted curves where training policies are sparse. The double spline scored lower than the quadratic on the fixed test set, but after refitting without two high-`BonusMalus` training policies, the spline's predictions for those policies rose to about 145 and 64 times their original values. The quadratic predictions changed by less than 2%, so the team retained it. Earlier test-tail diagnostics informed that choice. An independent sample would be needed to validate the choice without that influence. [Tail stress test](results/parametric_glm_notes.md#two-policy-tail-stress-test) · [Sensitivity plot](results/parametric_glm_bonus_malus.png)

### What did LightGBM tuning change?

Initial LightGBM used a learning rate of 0.05 and 300 estimators. Tuning compared six configurations using three folds within the training sample. The selected learning rate of 0.03 with 300 estimators lowered mean CV deviance from 74.509 to 73.831. It also lowered fixed-test deviance from 74.7513 to 74.2916 and narrowed the train–test gap from 10.174 to 7.879. [Baseline code](analyses/lgbm_baseline.py) · [Tuning code](analyses/lgbm_tuning.py) · [Tuning notes](results/lgbm_tuning_notes.md) · [Fold-level scores](results/lgbm_tuning_cv.json)

### What does the four-model comparison show?

Tuned LightGBM has the lowest test deviance and the highest observed Gini, with a small Gini improvement over initial LightGBM. Both GLMs predict portfolio totals within 1% of observed clipped test claims, while the initial and tuned LightGBM totals are about 17% and 14% below observed claims. Agreement on the total alone leaves calibration across risk groups open. [Full comparison and limitations](results/final_comparison.md) · [Test Lorenz curves](results/lorenz_comparison.png)

## Repository map

| Path | Contents |
| --- | --- |
| [`ps3/`](ps3/) | Data preparation, fixed split and preprocessing |
| [`analyses/`](analyses/) | The [original exercise sequence](analyses/ps3_script.py) and separate model implementations |
| [`tests/`](tests/) | Unit and integration tests |
| [`results/`](results/) | Submitted metrics, figures and analysis notes |

The original division of work is recorded in the [team assignments](TEAM_ASSIGNMENTS.md).

## PS4 — Exercise 1: monotonicity constraints

See the [PS4 environment and data setup](PS4_SETUP.md) for Conda setup, the shared data cache, and validation commands.

PS4 continues in this repository. The Ex1 implementation is in [`analyses/lgbm_monotonicity.py`](analyses/lgbm_monotonicity.py); it uses the PS3 fixed `IDpol` split, `ClaimAmountCut / Exposure` target, Exposure weights and Tweedie power 1.5. Three-fold shuffled KFold model selection is confined to the training sample. Each fold fits its own one-hot preprocessing, then the implementation resolves `BonusMalus` from the transformed feature names and applies the increasing constraint `+1` to that column. All remaining transformed columns receive constraint `0`.

Run after the PS3 Conda environment is installed. By default this loads the instructor's data and creates the fixed split:

```bash
python -m analyses.lgbm_monotonicity --results-dir results
```

To reuse a local prepared Parquet cache, pass `--data data/shared_final.parquet`. Both entry points recalculate `PurePremium` from `ClaimAmountCut / Exposure` and recreate the `IDpol` split, so older caches with stale target or split columns still use the canonical prepared data. The integrated entry point is `python -m analyses.ps3_script ps4-ex1`; it can also take `--data data/shared_final.parquet`. The run writes:

- `results/bonusmalus_weighted_claims.png` and its grouped CSV: training-only Exposure-weighted pure premium by observed `BonusMalus`.
- `results/ps4_ex1_monotonicity_cv.json`: fold scores for constrained and unconstrained parameter candidates and each selected configuration.
- `results/ps4_ex1_monotonicity_metrics.json`: split fingerprint, metrics, feature order, constraint vector and inputs for the later exercises.
- `results/ps4_ex1_test_predictions.csv`: test predictions aligned by `IDpol`, with `pp_t_lgbm_constrained` and `pp_t_lgbm_unconstrained`. This per-policy file is gitignored; other members can regenerate it with the command above.

The comparison reports selected CV deviance, train deviance and fixed-test deviance for both models. The test values are for final comparison only and do not participate in parameter selection.

### Ex1 result

Both model variants selected `learning_rate=0.03` and `n_estimators=300` from the PS3 tuning grid. Lower weighted Tweedie deviance is better.

| Model | Train-only CV mean | CV SD | Train | Fixed test |
| --- | ---: | ---: | ---: | ---: |
| Unconstrained LightGBM | 73.8312 | 2.4743 | 66.4130 | 74.2916 |
| Increasing BonusMalus constraint | 74.0892 | 2.3599 | 67.3737 | 74.4695 |

The constraint slightly worsened mean CV deviance by 0.2581 and fixed-test deviance by 0.1778, while narrowing the train–test gap from 7.8787 to 7.0957. Thus it imposes the required monotonic direction with a modest score tradeoff and a smaller apparent generalisation gap. The plot is descriptive; its 10-point bands have unequal support, especially in the high-BonusMalus tail. Policy counts and total Exposure are in the companion CSV.

Without the monotonicity constraint, a fitted LightGBM can predict a lower pure premium for a higher `BonusMalus` when the other predictors are held fixed. Tree split choices, interactions with other predictors, and sampling noise can create these local reversals, particularly where training exposure is sparse. The increasing constraint rules out decreases along `BonusMalus` while the other features remain fixed.

## PS4 — Exercise 2: Learning Curve

The constrained LightGBM was refitted using the parameters selected in Exercise 1: `learning_rate=0.03` and `n_estimators=300`.

The fixed IDpol split, training-fitted preprocessing, increasing BonusMalus constraint and Exposure weights were retained. Both training and test datasets were passed to `eval_set`, and the learning curve was generated using `lgb.plot_metric()`.

### Reproduce

Run Exercise 2 using the committed Exercise 1 report:

```bash
python -m analyses.ps3_script ps4-ex2 --data data/ps4_claims.parquet --results-dir results
```

By default, Exercise 2 reads `ps4_ex1_monotonicity_metrics.json` from the specified results directory.

To reproduce both exercises using a separate results directory:

```bash
python -m analyses.ps3_script ps4-ex1 --data data/ps4_claims.parquet --results-dir data/custom_results

python -m analyses.ps3_script ps4-ex2 --data data/ps4_claims.parquet --results-dir data/custom_results
```

Alternatively, an existing Exercise 1 report can be supplied explicitly:

```bash
python -m analyses.ps3_script ps4-ex2 --data data/ps4_claims.parquet --results-dir data/ex2_outputs --ex1-metrics-path results/ps4_ex1_monotonicity_metrics.json
```

The report must match the current dataset and fixed sample split. The learning curve uses one-based boosting rounds, consistent with the saved evaluation history and reported iteration numbers.

### Results

| Metric | Value |
|---|---:|
| Boosting iterations | 300 |
| Minimum test metric iteration | 110 |
| Minimum test metric | 45.169876 |
| Final training metric | 41.674999 |
| Final test metric | 45.292159 |
| Training Tweedie deviance | 67.373712 |
| Test Tweedie deviance | 74.469452 |

The training metric continued to decrease throughout the boosting iterations. The test metric reached its minimum at iteration 110 and then gradually increased, suggesting mild overfitting.

The original 300 estimators were retained because they were selected through training-only cross-validation. The test curve was used for diagnosis only, not for early stopping or parameter selection.

### Files

- [Learning curve implementation](analyses/learning_curve.py)
- [Learning curve figure](results/learning_curve.png)
- [Evaluation history and metrics](results/learning_curve_metrics.json)
- [Results and interpretation](results/learning_curve_notes.md)
- [Unit tests](tests/test_learning_curve.py)

## PS4 — Exercise 4: Partial dependence plots

From the repository root in the [shared `ps3` environment](PS4_SETUP.md):

```bash
python -m analyses.ps3_script ps4-ex4 --data data/ps4_claims.parquet --results-dir data/ps4_ex4 --ex1-metrics-path results/ps4_ex1_monotonicity_metrics.json
```

Without `--ex1-metrics-path`, the command looks for the Ex1 report inside `--results-dir`. The [PDP analysis](analyses/pdp_comparison.py) refits both Ex1 LightGBM pipelines without repeating cross-validation. It uses a shared sample of 100 training policies and shared feature grids to compute DALEX partial dependence for all nine features. The command writes interactive HTML plots, the profile CSVs, and the settings JSON. The checked-in [numeric](figures/ps4_ex4_pdp/numeric_pdp.png) and [categorical](figures/ps4_ex4_pdp/categorical_pdp.png) PNGs are snapshots of the comparison.

On the 21-point `BonusMalus` grid, the constrained profile never decreases, while the unconstrained profile has ten decreases. The `Density` profiles are comparatively close. These are model averages over the sampled background, not causal effects.

## PS4 — Exercise 5: Shapley decompositions

Run from the repository root in the [shared `ps3` environment](PS4_SETUP.md):

```bash
python -m analyses.ps3_script ps4-ex5 --data data/ps4_claims.parquet --results-dir results
```

The [analysis](analyses/shapley_decomposition.py) refits the initial Tweedie GLM and the constrained LightGBM on the same fixed training split, with `PurePremium = ClaimAmountCut / Exposure` as the target and `Exposure` as the training weight. It checks the split against the Exercise 1 report and reuses its selected LightGBM parameters and increasing `BonusMalus` constraint without repeating cross-validation. When writing to a separate directory, pass `--ex1-metrics-path results/ps4_ex1_monotonicity_metrics.json` to use the committed report.

Both DALEX explainers use the same 1,000 training policies sampled with seed 42. They evaluate the first test policy (`IDpol=13`) with `predict_parts(type="shap")`, averaging over 25 feature orderings with the same seed. The [interactive plot](results/ps4_ex5_shapley.html) shows both decompositions on a common scale (download the HTML and open it in a browser with internet access). The [JSON summary](results/ps4_ex5_shapley.json) contains the full numerical results.

| Model | Background baseline | Sum of SHAP contributions | Predicted pure premium |
| --- | ---: | ---: | ---: |
| Initial GLM | 146.09 | −60.19 | 85.89 |
| Constrained LightGBM | 129.59 | −39.58 | 90.00 |

The baseline is each model's unweighted mean prediction on the shared background, not the observed mean or the GLM intercept. The models can therefore have different baselines even though the background policies are identical. LightGBM starts about 16.50 lower but makes a net downward adjustment about 20.61 smaller than the GLM's, leaving its final prediction only 4.11 higher.

For this policy, the largest differences are in `Area` and `Density`. `Area=E` contributes +20.23 in the GLM but only +3.77 in LightGBM; `Density=3003` contributes +0.58 and +10.66, respectively. `BonusMalus=50` is the largest downward contribution in both models (−48.71 versus −39.19), and `Region=R31` and `VehBrand=B12` also pull the GLM prediction down more strongly. All nine contributions have the same sign in both models; their magnitudes differ. The GLM uses terms that are additive on the log scale, whereas tree splits can represent thresholds and interactions. This helps explain why the same feature values can change the two models' predictions by different amounts, but the SHAP results alone do not identify which split or interaction caused a difference. These are local attributions on the pure-premium scale, not GLM coefficients or evidence of global feature importance.
