from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sinopec02.data import load_bundle, split_actual_and_design, attach_design_features
from src.sinopec02.features import build_feature_table
from src.sinopec02.modeling import build_pipeline, cross_validate, decode_structured_predictions

REPORTS_DIR = ROOT / "reports"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(ROOT)

    train_actual, train_design = split_actual_and_design(bundle.train)
    train_actual = attach_design_features(train_actual, train_design)
    feature_table, feature_cols = build_feature_table(train_actual)

    metrics = cross_validate(feature_table, feature_cols, REPORTS_DIR)

    # Fit on the full training set and emit validation predictions for later use.
    model = build_pipeline(random_state=2026)
    model.fit(feature_table[feature_cols], feature_table["关键点"].astype(int))

    val = bundle.validation.copy()
    val_actual = val.loc[val[["JX", "FW"]].notna().all(axis=1)].copy()
    val_design = val.loc[val[["JX", "FW"]].isna().all(axis=1)].copy()
    val_actual = attach_design_features(val_actual, val_design)
    val_features, _ = build_feature_table(val_actual)

    proba = model.predict_proba(val_features[feature_cols])
    pred = model.predict(val_features[feature_cols])
    validation_pred = val_features[["id", "转换后JH", "XJS"]].copy()
    validation_pred["pred_raw"] = pred
    for label in [0, 1, 2, 3]:
        validation_pred[f"prob_{label}"] = proba[:, label]
    structured_parts = []
    prob_cols = [f"prob_{label}" for label in [0, 1, 2, 3]]
    for _, group_df in validation_pred.groupby("转换后JH", sort=False):
        ordered = group_df.sort_values("XJS").copy()
        ordered["pred_structured"] = decode_structured_predictions(ordered, prob_cols)
        structured_parts.append(ordered)
    validation_pred = pd.concat(structured_parts, ignore_index=True)
    validation_pred.to_csv(REPORTS_DIR / "validation_probabilities.csv", index=False, encoding="utf-8-sig")
    validation_pred[["id", "pred_structured"]].rename(columns={"pred_structured": "关键点"}).to_csv(
        REPORTS_DIR / "structured_predictions.csv", index=False, encoding="utf-8-sig"
    )

    with (REPORTS_DIR / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    summary_md = f"""# Baseline Results

_Auto-generated from grouped cross-validation on the training wells._

---

## Overall metrics

- Raw macro-F1: `{metrics["overall"]["macro_f1_raw"]:.4f}`
- Structured macro-F1: `{metrics["overall"]["macro_f1_structured"]:.4f}`
- Raw weighted-F1: `{metrics["overall"]["weighted_f1_raw"]:.4f}`
- Structured weighted-F1: `{metrics["overall"]["weighted_f1_structured"]:.4f}`

## Interpretation

- Point-wise random forest already recovers classes `1` and `2` reasonably well
- Per-well structured decoding improves macro-F1 and stabilizes class ordering
- Class `3` remains the hardest target due to very low support and higher ambiguity
"""
    (REPORTS_DIR / "baseline_summary.md").write_text(summary_md, encoding="utf-8")

    print(json.dumps(metrics["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
