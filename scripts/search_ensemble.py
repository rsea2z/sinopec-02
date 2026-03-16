from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sinopec02.data import LABEL_COL, WELL_COL
from src.sinopec02.modeling import decode_structured_predictions


COMPARE_DIR = ROOT / "reports" / "model_compare"
MODELS = ["random_forest", "extra_trees", "catboost"]


def load_oof(model_name: str) -> pd.DataFrame:
    return pd.read_csv(COMPARE_DIR / f"{model_name}_oof_predictions.csv")


def build_blend(weights: dict[str, float]) -> pd.DataFrame:
    base = None
    for model_name, weight in weights.items():
        df = load_oof(model_name).sort_values("id").reset_index(drop=True)
        proba_cols = [f"prob_{i}" for i in range(4)]
        if base is None:
            base = df[["id", WELL_COL, "XJS", LABEL_COL]].copy()
            for col in proba_cols:
                base[col] = 0.0
        for col in proba_cols:
            base[col] += weight * df[col]
    return base


def evaluate_blend(df: pd.DataFrame) -> tuple[float, float]:
    raw_pred = df[[f"prob_{i}" for i in range(4)]].to_numpy().argmax(axis=1)
    raw_macro = f1_score(df[LABEL_COL].astype(int), raw_pred, average="macro")

    structured_parts = []
    for _, group_df in df.groupby(WELL_COL, sort=False):
        ordered = group_df.sort_values("XJS").copy()
        ordered["pred_structured"] = decode_structured_predictions(
            ordered, [f"prob_{i}" for i in range(4)]
        )
        structured_parts.append(ordered)
    structured = pd.concat(structured_parts, ignore_index=True)
    structured_macro = f1_score(
        structured[LABEL_COL].astype(int), structured["pred_structured"].astype(int), average="macro"
    )
    return raw_macro, structured_macro


def main() -> None:
    rows = []
    candidates = [
        {"random_forest": 0.5, "extra_trees": 0.5},
        {"random_forest": 0.6, "extra_trees": 0.4},
        {"random_forest": 0.7, "extra_trees": 0.3},
        {"random_forest": 0.5, "extra_trees": 0.3, "catboost": 0.2},
        {"random_forest": 0.6, "extra_trees": 0.2, "catboost": 0.2},
        {"random_forest": 0.4, "extra_trees": 0.4, "catboost": 0.2},
    ]

    for weights in candidates:
        blend = build_blend(weights)
        raw_macro, structured_macro = evaluate_blend(blend)
        rows.append(
            {
                "weights": weights,
                "raw_macro_f1": raw_macro,
                "structured_macro_f1": structured_macro,
            }
        )

    result = pd.DataFrame(rows).sort_values("structured_macro_f1", ascending=False).reset_index(drop=True)
    result.to_csv(COMPARE_DIR / "ensemble_search.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

