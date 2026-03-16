# SINOPEC-02 项目记忆

_更新日期：2026-03-16_

---

## 当前任务

- 完成 `SINOPEC-02` 数据集的任务分析
- 建立可复现的 EDA、baseline、多模型比较与 ensemble 流程
- 在此基础上继续推进更强的序列模型 baseline

## 已确认事实

- 工作目录初始只有 `SINOPEC-02` 数据目录，已补齐为完整项目仓库
- `train.csv` 混合了实际轨迹点与设计轨迹点
- 设计轨迹点不在 `sample_submission.csv` 预测范围内
- 训练井与验证井不重合
- 标签顺序在井内稳定为 `1 -> 2 -> 3`
- `3` 类样本更少，部分井不存在 `3`
- `torch 2.7.0+cu128` 可用
- `pytorch_lightning` 当前环境存在依赖冲突，不适合作为本项目主训练框架

## 已落地内容

- 项目结构：`src/`、`scripts/`、`docs/analysis/`、`reports/`、`memory/`
- EDA 脚本：`scripts/eda_report.py`
- baseline 脚本：`scripts/train_baseline.py`
- 模型比较脚本：`scripts/compare_models.py`
- ensemble 搜索脚本：`scripts/search_ensemble.py`
- ensemble 预测脚本：`scripts/train_ensemble.py`
- 核心代码：
  - `src/sinopec02/data.py`
  - `src/sinopec02/features.py`
  - `src/sinopec02/modeling.py`
- 文档：
  - `README.md`
  - `docs/analysis/task_analysis.md`
  - `docs/analysis/model_plan.md`
  - `docs/analysis/baseline_results.md`
  - `docs/analysis/model_comparison.md`

## 当前结果

- 单模型最佳：`random_forest`
- 5 折按井交叉验证：
  - raw macro-F1: `0.5932`
  - structured macro-F1: `0.6322`
  - raw weighted-F1: `0.9803`
  - structured weighted-F1: `0.9823`
- 轻量 `BiLSTM` 序列模型：
  - raw macro-F1: `0.4292`
  - structured macro-F1: `0.5669`
  - 当前弱于表格模型与 ensemble

## 扩展实验结果

- 单模型比较已完成：
  - `random_forest`: `0.6322`
  - `extra_trees`: `0.6294`
  - `catboost`: `0.6267`
  - `lightgbm`: `0.5698`
- OOF 概率融合最佳组合：
  - `random_forest 0.6`
  - `extra_trees 0.2`
  - `catboost 0.2`
- ensemble structured macro-F1: `0.6414`
- 两阶段候选点模型：
  - threshold_2: `0.10`
  - threshold_3: `0.35`
  - macro-F1: `0.6643`
  - 当前为项目最优结果

## 关键建模决策

- 仅对实际轨迹点建模
- 设计轨迹通过 `XJS` 插值对齐到实际轨迹
- `FW` 使用圆周差与 `sin/cos` 编码
- 验证策略使用 `GroupKFold`
- baseline 采用点级分类加每井结构化解码
- 继续推进时优先尝试轻量纯 PyTorch 序列模型，而不是修复 Lightning 环境

## 当前阶段判断

- 数据理解：完成
- 可复现 baseline：完成
- 对比实验：完成
- 初步改进：完成
- 更强序列模型：进行中
- 轻量序列模型首轮：已完成，效果不如 ensemble
- 两阶段候选点路线：已完成可行性验证
- 两阶段候选点模型：已完成首轮实现并超过 ensemble
- 论文成稿：未开始

## 下一步

1. 深化两阶段候选点路线
2. 对 `3` 类做单独增强
3. 尝试候选点排序学习或联合解码
4. 当前候选覆盖率结论：
   - `1` 类 top-2: `95.3%`
   - `2` 类 top-5: `92.1%`
   - `3` 类 top-10: `92.0%`
