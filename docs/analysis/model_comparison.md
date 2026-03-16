# SINOPEC-02 模型比较

_基于同一套特征和按井交叉验证，对多种经典模型与集成方案进行比较。_

---

## 单模型结果

- `random_forest`: structured macro-F1 `0.6322`
- `extra_trees`: structured macro-F1 `0.6294`
- `catboost`: structured macro-F1 `0.6267`
- `lightgbm`: structured macro-F1 `0.5698`

## 集成结果

最佳概率融合组合为：

- `random_forest`: `0.6`
- `extra_trees`: `0.2`
- `catboost`: `0.2`

对应结构化 macro-F1：

- `0.6414`

## 解释

- `RandomForest` 仍然是最强单模型
- `ExtraTrees` 与 `CatBoost` 虽然单模略弱，但和 `RandomForest` 的错误分布并不完全相同
- 因此简单加权平均概率后，结构化解码能得到进一步提升

## 产物位置

- `reports/model_compare/model_summary.csv`
- `reports/model_compare/model_comparison.md`
- `reports/model_compare/ensemble_search.csv`
- `reports/model_compare/figures/model_comparison_macro_f1.png`
- `reports/ensemble/structured_predictions.csv`

