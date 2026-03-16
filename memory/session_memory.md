# SINOPEC-02 项目记忆

_更新日期：2026-03-16_

---

## 当前任务

- 完成 `SINOPEC-02` 数据集的任务分析
- 建立可复现的 EDA 与 baseline 流程
- 将项目内容推送到 `https://github.com/rsea2z/sinopec-02.git`

## 已确认事实

- 工作目录当前只有 `SINOPEC-02` 数据目录，初始时不是 git 仓库
- `train.csv` 混合了实际轨迹点与设计轨迹点
- 设计轨迹点不在 `sample_submission.csv` 预测范围内
- 训练井与验证井不重合
- 标签顺序在井内稳定为 `1 -> 2 -> 3`
- `3` 类样本明显更少，部分井不存在 `3`

## 已落地内容

- 项目结构：`src/`、`scripts/`、`docs/analysis/`、`reports/`、`memory/`
- EDA 脚本：`scripts/eda_report.py`
- baseline 脚本：`scripts/train_baseline.py`
- 核心代码：
  - `src/sinopec02/data.py`
  - `src/sinopec02/features.py`
  - `src/sinopec02/modeling.py`
- 文档：
  - `README.md`
  - `docs/analysis/task_analysis.md`
  - `docs/analysis/model_plan.md`
  - `docs/analysis/baseline_results.md`

## 当前结果

- 5 折按井交叉验证
- raw macro-F1: `0.5932`
- structured macro-F1: `0.6322`
- raw weighted-F1: `0.9803`
- structured weighted-F1: `0.9823`
- 结构化解码对 `1`、`2` 类提升明显
- `3` 类仍是主要瓶颈

## 关键建模决策

- 仅对实际轨迹点建模
- 设计轨迹通过 `XJS` 插值对齐到实际轨迹
- `FW` 使用圆周差与 `sin/cos` 编码
- 验证策略使用 `GroupKFold`
- baseline 采用点级随机森林加每井结构化解码

## 下一步

1. 运行脚本生成报告与指标
2. 初始化 git，提交并推送到目标仓库
3. 如需继续，尝试更强 baseline 与两阶段方法
