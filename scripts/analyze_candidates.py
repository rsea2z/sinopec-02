from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sinopec02.data import LABEL_COL, WELL_COL


COMPARE_DIR = ROOT / "reports" / "model_compare"
OUTPUT_DIR = ROOT / "reports" / "candidate_analysis"
WEIGHTS = {"random_forest": 0.6, "extra_trees": 0.2, "catboost": 0.2}


def load_oof(model_name: str) -> pd.DataFrame:
    return pd.read_csv(COMPARE_DIR / f"{model_name}_oof_predictions.csv").sort_values("id").reset_index(drop=True)


def build_ensemble_oof() -> pd.DataFrame:
    base = None
    for model_name, weight in WEIGHTS.items():
        df = load_oof(model_name)
        if base is None:
            base = df[["id", WELL_COL, "XJS", LABEL_COL]].copy()
            for label in [0, 1, 2, 3]:
                base[f"prob_{label}"] = 0.0
        for label in [0, 1, 2, 3]:
            base[f"prob_{label}"] += weight * df[f"prob_{label}"]
    return base


def coverage_at_k(df: pd.DataFrame, label: int, k: int) -> float:
    hit = 0
    total = 0
    for _, group_df in df.groupby(WELL_COL, sort=False):
        truth_rows = group_df[group_df[LABEL_COL] == label]
        if truth_rows.empty:
            continue
        total += len(truth_rows)
        top_ids = set(group_df.nlargest(k, f"prob_{label}")["id"])
        hit += truth_rows["id"].isin(top_ids).sum()
    return hit / total if total else 0.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oof = build_ensemble_oof()

    rows = []
    for label in [1, 2, 3]:
        for k in [1, 2, 3, 5, 8, 10]:
            rows.append(
                {
                    "label": label,
                    "k": k,
                    "coverage": coverage_at_k(oof, label, k),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "candidate_coverage.csv", index=False, encoding="utf-8-sig")
    with (OUTPUT_DIR / "candidate_coverage.json").open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    pivot = result.pivot(index="k", columns="label", values="coverage")
    md = "# Candidate Coverage\n\n" + pivot.to_markdown() + "\n"
    (OUTPUT_DIR / "candidate_coverage.md").write_text(md, encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

