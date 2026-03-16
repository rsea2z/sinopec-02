from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sinopec02.data import attach_design_features, load_bundle, split_actual_and_design
from src.sinopec02.features import build_feature_table
from src.sinopec02.modeling import cross_validate


REPORTS_DIR = ROOT / "reports" / "model_compare"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS = ["random_forest", "extra_trees", "lightgbm", "catboost"]


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def build_training_frame() -> tuple[pd.DataFrame, list[str]]:
    bundle = load_bundle(ROOT)
    train_actual, train_design = split_actual_and_design(bundle.train)
    train_actual = attach_design_features(train_actual, train_design)
    return build_feature_table(train_actual)


def plot_model_scores(df: pd.DataFrame) -> None:
    ordered = df.sort_values("macro_f1_structured", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(ordered["model_name"], ordered["macro_f1_structured"], color="#2563eb", alpha=0.85, label="structured")
    ax.scatter(ordered["model_name"], ordered["macro_f1_raw"], color="#ea580c", s=60, zorder=3, label="raw")
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Grouped CV model comparison")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for idx, row in ordered.iterrows():
        ax.text(idx, row["macro_f1_structured"] + 0.005, f'{row["macro_f1_structured"]:.3f}', ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison_macro_f1.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    feature_table, feature_cols = build_training_frame()

    summaries = []
    for model_name in MODELS:
        metrics = cross_validate(
            feature_table,
            feature_cols,
            REPORTS_DIR,
            model_name=model_name,
            output_prefix=model_name,
        )
        overall = metrics["overall"]
        summaries.append(
            {
                "model_name": model_name,
                "macro_f1_raw": overall["macro_f1_raw"],
                "macro_f1_structured": overall["macro_f1_structured"],
                "weighted_f1_raw": overall["weighted_f1_raw"],
                "weighted_f1_structured": overall["weighted_f1_structured"],
            }
        )

    summary_df = pd.DataFrame(summaries).sort_values("macro_f1_structured", ascending=False)
    summary_df.to_csv(REPORTS_DIR / "model_summary.csv", index=False, encoding="utf-8-sig")
    with (REPORTS_DIR / "model_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    plot_model_scores(summary_df)

    best = summary_df.iloc[0].to_dict()
    md = f"""# Model Comparison

_Auto-generated grouped cross-validation comparison across multiple classical models._

---

## Ranking

{summary_df.to_markdown(index=False)}

## Best model

- Model: `{best["model_name"]}`
- Structured macro-F1: `{best["macro_f1_structured"]:.4f}`
- Raw macro-F1: `{best["macro_f1_raw"]:.4f}`
"""
    (REPORTS_DIR / "model_comparison.md").write_text(md, encoding="utf-8")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

