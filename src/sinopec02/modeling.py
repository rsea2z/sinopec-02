from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from .data import LABEL_COL, WELL_COL


LABELS = [0, 1, 2, 3]


@dataclass
class FoldResult:
    fold: int
    macro_f1_raw: float
    macro_f1_structured: float
    weighted_f1_raw: float
    weighted_f1_structured: float


def build_pipeline(random_state: int = 42) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def decode_structured_predictions(group: pd.DataFrame, prob_cols: list[str]) -> np.ndarray:
    arr = group[prob_cols].to_numpy(dtype=float)
    n = arr.shape[0]
    pred = np.zeros(n, dtype=int)

    score_1 = arr[:, 1]
    score_2 = arr[:, 2]
    score_3 = arr[:, 3]

    best_value = float("-inf")
    best_triplet: tuple[int | None, int | None, int | None] = (None, None, None)

    # Allow label 3 to be absent because many wells do not contain a drop-off point.
    for i in range(n):
        for j in range(i, n):
            base = score_1[i] + score_2[j]
            if base > best_value:
                best_value = base
                best_triplet = (i, j, None)
            for k in range(j, n):
                total = base + score_3[k]
                if total > best_value:
                    best_value = total
                    best_triplet = (i, j, k)

    i, j, k = best_triplet
    if i is not None:
        pred[i] = 1
    if j is not None:
        pred[j] = 2
    if k is not None:
        pred[k] = 3
    return pred


def cross_validate(
    data: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path | str,
    n_splits: int = 5,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = data[feature_cols]
    y = data[LABEL_COL].astype(int)
    groups = data[WELL_COL]

    splitter = GroupKFold(n_splits=n_splits)
    oof_records: list[pd.DataFrame] = []
    fold_results: list[FoldResult] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = build_pipeline(random_state=42 + fold)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        raw_pred = model.predict(X.iloc[valid_idx])
        proba = model.predict_proba(X.iloc[valid_idx])
        proba_cols = [f"prob_{label}" for label in LABELS]

        fold_df = data.iloc[valid_idx][["id", WELL_COL, "XJS", LABEL_COL]].copy()
        fold_df["pred_raw"] = raw_pred
        for i, label in enumerate(LABELS):
            fold_df[f"prob_{label}"] = proba[:, i]

        structured_pred_parts: list[pd.DataFrame] = []
        for _, group_df in fold_df.groupby(WELL_COL, sort=False):
            ordered = group_df.sort_values("XJS").copy()
            ordered["pred_structured"] = decode_structured_predictions(ordered, proba_cols)
            structured_pred_parts.append(ordered)

        fold_df = pd.concat(structured_pred_parts, ignore_index=True)
        oof_records.append(fold_df)

        fold_results.append(
            FoldResult(
                fold=fold,
                macro_f1_raw=f1_score(fold_df[LABEL_COL], fold_df["pred_raw"], average="macro"),
                macro_f1_structured=f1_score(
                    fold_df[LABEL_COL], fold_df["pred_structured"], average="macro"
                ),
                weighted_f1_raw=f1_score(
                    fold_df[LABEL_COL], fold_df["pred_raw"], average="weighted"
                ),
                weighted_f1_structured=f1_score(
                    fold_df[LABEL_COL], fold_df["pred_structured"], average="weighted"
                ),
            )
        )

    oof = pd.concat(oof_records, ignore_index=True).sort_values("id").reset_index(drop=True)
    metrics = summarize_metrics(oof, fold_results)

    oof.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "cv_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def summarize_metrics(oof: pd.DataFrame, fold_results: list[FoldResult]) -> dict:
    y_true = oof[LABEL_COL].astype(int)
    y_pred_raw = oof["pred_raw"].astype(int)
    y_pred_structured = oof["pred_structured"].astype(int)

    return {
        "fold_results": [result.__dict__ for result in fold_results],
        "overall": {
            "macro_f1_raw": f1_score(y_true, y_pred_raw, average="macro"),
            "macro_f1_structured": f1_score(y_true, y_pred_structured, average="macro"),
            "weighted_f1_raw": f1_score(y_true, y_pred_raw, average="weighted"),
            "weighted_f1_structured": f1_score(y_true, y_pred_structured, average="weighted"),
            "classification_report_raw": classification_report(
                y_true, y_pred_raw, labels=LABELS, output_dict=True, zero_division=0
            ),
            "classification_report_structured": classification_report(
                y_true, y_pred_structured, labels=LABELS, output_dict=True, zero_division=0
            ),
            "confusion_matrix_raw": confusion_matrix(y_true, y_pred_raw, labels=LABELS).tolist(),
            "confusion_matrix_structured": confusion_matrix(
                y_true, y_pred_structured, labels=LABELS
            ).tolist(),
        },
    }
