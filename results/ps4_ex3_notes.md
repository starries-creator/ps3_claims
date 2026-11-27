# PS4 Exercise 3 — Evaluation of the LightGBM variants

The fixed test set contains 135,727 policies. Both models were refitted on the
542,286 canonical training policies using their independently selected Ex1
parameters and the Ex1 preprocessing, without repeating cross-validation. The
constrained model has a +1 BonusMalus constraint and zero constraints on the
other transformed features. Predictions were matched one-to-one by `IDpol`.
The exact split fingerprint and complete settings are recorded in the JSON.

| Metric | Unconstrained | Constrained |
| --- | ---: | ---: |
| Relative bias | -0.143084 | -0.124451 |
| Tweedie deviance | 74.291639 | 74.469452 |
| MAE | 242.544795 | 245.649803 |
| RMSE | 3599.426751 | 3599.604825 |
| Gini | 0.274197 | 0.272074 |

Relative bias measures aggregate calibration: both models underpredict total
claims, by approximately 14.31% and 12.45%, respectively. MAE measures the
exposure-weighted average absolute error, while RMSE squares errors and is more
sensitive to large misses. Tweedie deviance with power 1.5 is appropriate for
nonnegative pure premium with a mass at zero and continuous positive claims.
The unconstrained model has lower deviance, MAE and RMSE; the constrained model
has smaller absolute aggregate bias. The constraint offers a more interpretable
nondecreasing BonusMalus effect, at a modest loss on these three error metrics.
The unnormalized PS3 Lorenz Gini measures ranking of claims across policies; the
unconstrained model ranks slightly better on this sample. These results do not
establish statistical significance or superiority on every criterion.

The recomputed Ex1 deviances differ from the full-precision handoff by about
1.42e-14 for each model (tolerance `rtol=1e-8`, `atol=1e-8`). The official
Notion handout was inaccessible during implementation; the bias definition and
other metrics follow the repository acceptance rubric and documented fallback.
