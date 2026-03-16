# SINOPEC-02 两阶段候选点结果

_记录基于候选点生成与候选点精排的两阶段实验结果。_

---

## 方法

### 第一阶段

使用当前最佳 ensemble 作为候选点生成器：

- `random_forest 0.6`
- `extra_trees 0.2`
- `catboost 0.2`

并按井生成候选集合：

- `1` 类：top-2
- `2` 类：top-5
- `3` 类：top-10

### 第二阶段

在候选集合内，为每个标签单独训练 `ExtraTrees` 二分类器，输入包括：

- 原始工程特征
- 设计对齐偏差特征
- 第一阶段概率
- 概率排序特征

最后做按井结构化解码，并搜索 `2/3` 类阈值。

## 结果

- threshold_2: `0.10`
- threshold_3: `0.35`
- max_rank_2: `2`
- max_rank_3: `3`
- two-stage macro-F1: `0.6656`
- stage-1 structured reference: `0.6414`

## 结论

- 两阶段方法已经明显超过当前最佳单阶段 ensemble
- 对 `2/3` 类候选排名做门控后，还能获得一小步但稳定的提升
- 候选点生成 + 候选点精排 是当前最值得继续深入的路线
- 下一步应继续优化：
  - 更好的 second-stage 特征
  - label-3 专项增强
  - 联合排序而非独立二分类

## 产物位置

- `reports/two_stage/two_stage_metrics.json`
- `reports/two_stage/two_stage_oof_predictions.csv`
- `reports/two_stage/two_stage_structured_predictions.csv`
- `reports/two_stage/two_stage_summary.md`
