from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sinopec02.data import WELL_COL, attach_design_features, load_bundle, split_actual_and_design
from src.sinopec02.features import build_feature_table
from src.sinopec02.modeling import build_pipeline, decode_structured_predictions


REPORTS_DIR = ROOT / "reports" / "ensemble"
WEIGHTS = {"random_forest": 0.6, "extra_trees": 0.2, "catboost": 0.2}


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(ROOT)

    train_actual, train_design = split_actual_and_design(bundle.train)
    train_actual = attach_design_features(train_actual, train_design)
    train_features, feature_cols = build_feature_table(train_actual)

    val = bundle.validation.copy()
    val_actual = val.loc[val[["JX", "FW"]].notna().all(axis=1)].copy()
    val_design = val.loc[val[["JX", "FW"]].isna().all(axis=1)].copy()
    val_actual = attach_design_features(val_actual, val_design)
    val_features, _ = build_feature_table(val_actual)

    pred_df = val_features[["id", WELL_COL, "XJS"]].copy()
    for label in [0, 1, 2, 3]:
        pred_df[f"prob_{label}"] = 0.0

    for model_name, weight in WEIGHTS.items():
        model = build_pipeline(model_name=model_name, random_state=2026)
        model.fit(train_features[feature_cols], train_features["关键点"].astype(int))
        proba = model.predict_proba(val_features[feature_cols])
        for label in [0, 1, 2, 3]:
            pred_df[f"prob_{label}"] += weight * proba[:, label]

    pred_df["pred_raw"] = pred_df[[f"prob_{i}" for i in range(4)]].to_numpy().argmax(axis=1)

    structured_parts = []
    for _, group_df in pred_df.groupby(WELL_COL, sort=False):
        ordered = group_df.sort_values("XJS").copy()
        ordered["pred_structured"] = decode_structured_predictions(
            ordered, [f"prob_{i}" for i in range(4)]
        )
        structured_parts.append(ordered)
    pred_df = pd.concat(structured_parts, ignore_index=True)

    pred_df.to_csv(REPORTS_DIR / "validation_probabilities.csv", index=False, encoding="utf-8-sig")
    pred_df[["id", "pred_structured"]].rename(columns={"pred_structured": "关键点"}).to_csv(
        REPORTS_DIR / "structured_predictions.csv", index=False, encoding="utf-8-sig"
    )
    with (REPORTS_DIR / "ensemble_config.json").open("w", encoding="utf-8") as f:
        json.dump({"weights": WEIGHTS, "feature_count": len(feature_cols)}, f, ensure_ascii=False, indent=2)

    print(json.dumps({"weights": WEIGHTS, "rows": len(pred_df)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

