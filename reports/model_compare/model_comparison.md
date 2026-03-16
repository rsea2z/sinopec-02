# Model Comparison

_Auto-generated grouped cross-validation comparison across multiple classical models._

---

## Ranking

| model_name    |   macro_f1_raw |   macro_f1_structured |   weighted_f1_raw |   weighted_f1_structured |
|:--------------|---------------:|----------------------:|------------------:|-------------------------:|
| random_forest |       0.593184 |              0.632166 |          0.980312 |                 0.982309 |
| extra_trees   |       0.574811 |              0.629426 |          0.977758 |                 0.981942 |
| catboost      |       0.560789 |              0.626711 |          0.972533 |                 0.981897 |
| lightgbm      |       0.53331  |              0.569757 |          0.981637 |                 0.98032  |

## Best model

- Model: `random_forest`
- Structured macro-F1: `0.6322`
- Raw macro-F1: `0.5932`
