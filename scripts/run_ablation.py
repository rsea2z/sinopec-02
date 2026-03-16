from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sinopec02.data import attach_design_features, load_bundle, split_actual_and_design
from src.sinopec02.features import build_feature_table
from src.sinopec02.modeling import cross_validate


REPORTS_DIR = ROOT / "reports" / "ablation"


def keep_feature_subset(feature_cols: list[str], mode: str) -> list[str]:
    if mode == "all_features":
        return feature_cols
    if mode == "no_design":
        return [c for c in feature_cols if "design" not in c]
    if mode == "no_dynamic":
        drop_tokens = ["dJX", "d2JX", "dFW", "dLJCZJS", "JX_minus_", "abs_dFW"]
        return [c for c in feature_cols if not any(tok in c for tok in drop_tokens)]
    if mode == "no_rolling":
        return [c for c in feature_cols if "_roll_" not in c]
    if mode == "no_position":
        return [c for c in feature_cols if c not in {"well_index", "well_length", "well_pos_frac"}]
    if mode == "raw_minimal":
        minimal = {
            "XJS",
            "JX",
            "FW",
            "LJCZJS",
            "FW_sin",
            "FW_cos",
            "JX_design_aligned",
            "FW_design_aligned",
            "LJCZJS_design_aligned",
            "delta_JX_design",
            "delta_FW_design",
            "delta_LJCZJS_design",
        }
        return [c for c in feature_cols if c in minimal]
    raise ValueError(mode)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(ROOT)
    train_actual, train_design = split_actual_and_design(bundle.train)
    train_actual = attach_design_features(train_actual, train_design)
    feature_table, feature_cols = build_feature_table(train_actual)

    configs = [
        "all_features",
        "no_design",
        "no_dynamic",
        "no_rolling",
        "no_position",
        "raw_minimal",
    ]

    rows = []
    for config in configs:
        subset = keep_feature_subset(feature_cols, config)
        metrics = cross_validate(
            feature_table,
            subset,
            REPORTS_DIR,
            model_name="random_forest",
            output_prefix=f"ablation_{config}",
        )
        rows.append(
            {
                "config": config,
                "feature_count": len(subset),
                "macro_f1_raw": metrics["overall"]["macro_f1_raw"],
                "macro_f1_structured": metrics["overall"]["macro_f1_structured"],
            }
        )

    df = pd.DataFrame(rows).sort_values("macro_f1_structured", ascending=False).reset_index(drop=True)
    df.to_csv(REPORTS_DIR / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    with (REPORTS_DIR / "ablation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    with (REPORTS_DIR / "ablation_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Ablation Summary\n\n" + df.to_markdown(index=False) + "\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

