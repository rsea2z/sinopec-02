from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sinopec02.data import (
    ACTUAL_NUMERIC_COLS,
    DESIGN_NUMERIC_COLS,
    LABEL_COL,
    WELL_COL,
    load_bundle,
    per_well_counts,
    split_actual_and_design,
)

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = ROOT / "docs" / "analysis"


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, FIGURES_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def plot_label_distribution(actual_train: pd.DataFrame) -> str:
    counts = actual_train[LABEL_COL].astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values, color=["#9ca3af", "#2563eb", "#16a34a", "#ea580c"])
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_title("Train Label Distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for x, y in zip(counts.index.astype(str), counts.values):
        ax.text(x, y, str(y), ha="center", va="bottom", fontsize=9)
    path = FIGURES_DIR / "label_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path.relative_to(ROOT)).replace("\\", "/")


def plot_well_lengths(actual_train: pd.DataFrame, actual_val: pd.DataFrame) -> str:
    train_counts = actual_train.groupby(WELL_COL).size()
    val_counts = actual_val.groupby(WELL_COL).size()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(train_counts, bins=12, alpha=0.7, label="train", color="#2563eb")
    ax.hist(val_counts, bins=12, alpha=0.7, label="validation", color="#16a34a")
    ax.set_xlabel("Actual trajectory points per well")
    ax.set_ylabel("Number of wells")
    ax.set_title("Sequence length distribution by well")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    path = FIGURES_DIR / "well_length_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path.relative_to(ROOT)).replace("\\", "/")


def build_summary(bundle) -> dict:
    train_actual, train_design = split_actual_and_design(bundle.train)
    val_actual_mask = bundle.validation[["JX", "FW"]].notna().all(axis=1)
    val_actual = bundle.validation.loc[val_actual_mask].copy()
    val_design = bundle.validation.loc[~val_actual_mask].copy()

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": int(len(bundle.train)),
        "validation_rows": int(len(bundle.validation)),
        "submission_rows": int(len(bundle.submission)),
        "train_actual_rows": int(len(train_actual)),
        "train_design_rows": int(len(train_design)),
        "validation_actual_rows": int(len(val_actual)),
        "validation_design_rows": int(len(val_design)),
        "train_wells": int(train_actual[WELL_COL].nunique()),
        "validation_wells": int(val_actual[WELL_COL].nunique()),
        "train_validation_well_overlap": int(
            len(set(train_actual[WELL_COL]) & set(val_actual[WELL_COL]))
        ),
        "label_distribution": {
            str(int(k)): int(v)
            for k, v in train_actual[LABEL_COL].astype(int).value_counts().sort_index().items()
        },
        "train_missing": bundle.train.isna().sum().to_dict(),
        "validation_missing": bundle.validation.isna().sum().to_dict(),
        "train_well_counts": per_well_counts(bundle.train).to_dict(orient="records"),
        "validation_well_counts": per_well_counts(bundle.validation).to_dict(orient="records"),
        "actual_numeric_summary": train_actual[ACTUAL_NUMERIC_COLS].describe().round(4).to_dict(),
        "design_numeric_summary": train_design[["XJS"] + DESIGN_NUMERIC_COLS].describe().round(4).to_dict(),
    }
    return summary


def write_markdown(summary: dict, label_fig: str, length_fig: str) -> None:
    report = f"""# SINOPEC-02 数据探索报告

_自动生成于 {summary["generated_at"]}，用于任务理解、建模约束确认与 baseline 设计。_

---

## 数据结构

- 训练集总行数：`{summary["train_rows"]}`
- 验证集总行数：`{summary["validation_rows"]}`
- 提交文件行数：`{summary["submission_rows"]}`
- 训练集实际轨迹点：`{summary["train_actual_rows"]}`
- 训练集设计轨迹点：`{summary["train_design_rows"]}`
- 验证集实际轨迹点：`{summary["validation_actual_rows"]}`
- 验证集设计轨迹点：`{summary["validation_design_rows"]}`

```mermaid
flowchart TB
    accTitle: Dataset Composition
    accDescr: This diagram shows how the train and validation tables each contain both actual trajectory rows and design trajectory rows, while the submission file only covers actual validation rows.

    train[📜 train.csv] --> train_actual[🧠 实际轨迹点]
    train --> train_design[📎 设计轨迹点]
    validation[📜 validation_without_label.csv] --> val_actual[🧠 实际轨迹点]
    validation --> val_design[📎 设计轨迹点]
    sample[📤 sample_submission.csv] --> val_actual

    classDef primary fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef success fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class train,validation,sample primary
    class train_actual,train_design,val_actual,val_design success
```

## 核心发现

- 训练井数量：`{summary["train_wells"]}`
- 验证井数量：`{summary["validation_wells"]}`
- 训练井与验证井重合数：`{summary["train_validation_well_overlap"]}`
- 结论：验证集为未见井，评估必须按井分组进行

- 标签分布：
  - `0`: `{summary["label_distribution"].get("0", 0)}`
  - `1`: `{summary["label_distribution"].get("1", 0)}`
  - `2`: `{summary["label_distribution"].get("2", 0)}`
  - `3`: `{summary["label_distribution"].get("3", 0)}`

![训练集关键点标签分布](../../{label_fig})

![按井统计的序列长度分布](../../{length_fig})

## 建模含义

- 该任务应视为按井分组的序列关键点识别问题
- 标签高度不平衡，不能只看整体准确率
- 设计轨迹与实际轨迹分开存储，使用前必须按 `XJS` 对齐
- `FW` 是圆周变量，需要采用角度差而非普通减法
- 同一口井内关键点满足顺序约束：`1 -> 2 -> 3`

## 推荐下一步

1. 仅保留实际轨迹点作为预测对象
2. 用设计轨迹作为辅助特征来源
3. 做按井 `GroupKFold` 交叉验证
4. 在点级分类之后加入每井结构化后处理
"""
    (DOCS_DIR / "data_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    bundle = load_bundle(ROOT)
    summary = build_summary(bundle)

    train_actual, _ = split_actual_and_design(bundle.train)
    val_actual = bundle.validation.loc[bundle.validation[["JX", "FW"]].notna().all(axis=1)].copy()
    label_fig = plot_label_distribution(train_actual)
    length_fig = plot_well_lengths(train_actual, val_actual)

    with (REPORTS_DIR / "data_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_markdown(summary, label_fig, length_fig)
    print("EDA report generated.")


if __name__ == "__main__":
    main()
