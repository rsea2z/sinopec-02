# SINOPEC-02 候选点分析

_验证两阶段候选点检测路线是否有足够的候选覆盖率。_

---

## 方法

基于当前最佳 ensemble 概率：

- `random_forest 0.6`
- `extra_trees 0.2`
- `catboost 0.2`

在训练集 OOF 概率上，按井分别取每个标签概率最高的 top-k 点，统计真实关键点是否落入候选集合。

## 结果

- `1` 类：
  - top-1 coverage: `0.7656`
  - top-2 coverage: `0.9531`
  - top-5 coverage: `0.9844`
- `2` 类：
  - top-1 coverage: `0.6349`
  - top-3 coverage: `0.8413`
  - top-5 coverage: `0.9206`
  - top-10 coverage: `0.9524`
- `3` 类：
  - top-1 coverage: `0.3200`
  - top-3 coverage: `0.6000`
  - top-5 coverage: `0.7600`
  - top-10 coverage: `0.9200`

## 结论

- `1` 类和 `2` 类非常适合先做候选点筛选，再做精排
- `3` 类较难，但只要候选池放宽到 top-10，覆盖率也能达到 `92%`
- 因此下一步做“两阶段候选点检测”是合理的

## 产物位置

- `reports/candidate_analysis/candidate_coverage.csv`
- `reports/candidate_analysis/candidate_coverage.json`
- `reports/candidate_analysis/candidate_coverage.md`

