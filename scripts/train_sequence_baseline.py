from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sinopec02.data import attach_design_features, load_bundle, split_actual_and_design
from src.sinopec02.features import build_feature_table
from src.sinopec02.sequence import cross_validate_sequence


REPORTS_DIR = ROOT / "reports" / "sequence"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(ROOT)
    train_actual, train_design = split_actual_and_design(bundle.train)
    train_actual = attach_design_features(train_actual, train_design)
    feature_table, feature_cols = build_feature_table(train_actual)
    metrics = cross_validate_sequence(feature_table, feature_cols, REPORTS_DIR)
    with (REPORTS_DIR / "sequence_summary.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Sequence Baseline Results\n\n"
            f"- Raw macro-F1: `{metrics['overall']['macro_f1_raw']:.4f}`\n"
            f"- Structured macro-F1: `{metrics['overall']['macro_f1_structured']:.4f}`\n"
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

