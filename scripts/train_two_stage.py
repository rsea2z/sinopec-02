from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sinopec02.data import LABEL_COL, WELL_COL, attach_design_features, load_bundle, split_actual_and_design
from src.sinopec02.features import build_feature_table


REPORTS_DIR = ROOT / "reports" / "two_stage"
MODEL_COMPARE_DIR = ROOT / "reports" / "model_compare"
ENSEMBLE_DIR = ROOT / "reports" / "ensemble"
WEIGHTS = {"random_forest": 0.6, "extra_trees": 0.2, "catboost": 0.2}
TOP_K = {1: 2, 2: 5, 3: 10}


def load_stage1_oof() -> pd.DataFrame:
    base = None
    for model_name, weight in WEIGHTS.items():
        df = pd.read_csv(MODEL_COMPARE_DIR / f"{model_name}_oof_predictions.csv").sort_values("id").reset_index(drop=True)
        if base is None:
            base = df[["id", WELL_COL, "XJS", LABEL_COL]].copy()
            for label in [0, 1, 2, 3]:
                base[f"stage1_prob_{label}"] = 0.0
        for label in [0, 1, 2, 3]:
            base[f"stage1_prob_{label}"] += weight * df[f"prob_{label}"]
    return base


def load_stage1_validation() -> pd.DataFrame:
    df = pd.read_csv(ENSEMBLE_DIR / "validation_probabilities.csv")
    df = df.rename(columns={f"prob_{i}": f"stage1_prob_{i}" for i in [0, 1, 2, 3]})
    return df[["id", WELL_COL, "XJS", "stage1_prob_0", "stage1_prob_1", "stage1_prob_2", "stage1_prob_3"]]


def prepare_feature_frames() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    bundle = load_bundle(ROOT)
    train_actual, train_design = split_actual_and_design(bundle.train)
    train_actual = attach_design_features(train_actual, train_design)
    train_features, feature_cols = build_feature_table(train_actual)

    val = bundle.validation.copy()
    val_actual = val.loc[val[["JX", "FW"]].notna().all(axis=1)].copy()
    val_design = val.loc[val[["JX", "FW"]].isna().all(axis=1)].copy()
    val_actual = attach_design_features(val_actual, val_design)
    val_features, _ = build_feature_table(val_actual)
    return train_features, val_features, feature_cols


def add_stage1_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for label in [1, 2, 3]:
        df[f"rank_prob_{label}"] = (
            df.groupby(WELL_COL)[f"stage1_prob_{label}"].rank(method="first", ascending=False)
        )
        df[f"inv_rank_prob_{label}"] = 1.0 / df[f"rank_prob_{label}"]
    df["stage1_nonzero_sum"] = df["stage1_prob_1"] + df["stage1_prob_2"] + df["stage1_prob_3"]
    df["stage1_best_nonzero"] = df[[f"stage1_prob_{i}" for i in [1, 2, 3]]].max(axis=1)
    return df


def select_candidates(df: pd.DataFrame, label: int) -> pd.DataFrame:
    parts = []
    prob_col = f"stage1_prob_{label}"
    for _, group_df in df.groupby(WELL_COL, sort=False):
        top = group_df.nlargest(TOP_K[label], prob_col).copy()
        top[f"is_candidate_{label}"] = 1
        parts.append(top)
    out = pd.concat(parts, ignore_index=True)
    out[f"target_label_{label}"] = (out[LABEL_COL] == label).astype(int) if LABEL_COL in out.columns else 0
    return out


def stage2_feature_columns(base_feature_cols: list[str]) -> list[str]:
    extra = [f"stage1_prob_{i}" for i in [0, 1, 2, 3]]
    extra += [f"rank_prob_{i}" for i in [1, 2, 3]]
    extra += [f"inv_rank_prob_{i}" for i in [1, 2, 3]]
    extra += ["stage1_nonzero_sum", "stage1_best_nonzero"]
    return base_feature_cols + extra


def fit_label_oof(cand_df: pd.DataFrame, label: int, feature_cols: list[str]) -> pd.DataFrame:
    cand_df = cand_df.copy().sort_values("id").reset_index(drop=True)
    groups = cand_df[WELL_COL]
    y = cand_df[f"target_label_{label}"].astype(int)
    splitter = GroupKFold(n_splits=5)
    scores = np.zeros(len(cand_df), dtype=float)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(cand_df[feature_cols], y, groups), start=1):
        model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=1,
            random_state=700 + label * 10 + fold,
        )
        model.fit(cand_df.iloc[train_idx][feature_cols].fillna(0.0), y.iloc[train_idx])
        scores[valid_idx] = model.predict_proba(cand_df.iloc[valid_idx][feature_cols].fillna(0.0))[:, 1]

    cand_df[f"stage2_score_{label}"] = scores
    return cand_df


def fit_label_full(train_cand_df: pd.DataFrame, pred_cand_df: pd.DataFrame, label: int, feature_cols: list[str]) -> pd.DataFrame:
    train_y = train_cand_df[f"target_label_{label}"].astype(int)
    model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=1,
        random_state=900 + label,
    )
    model.fit(train_cand_df[feature_cols].fillna(0.0), train_y)
    pred_cand_df = pred_cand_df.copy()
    pred_cand_df[f"stage2_score_{label}"] = model.predict_proba(pred_cand_df[feature_cols].fillna(0.0))[:, 1]
    return pred_cand_df


def decode_well(group_df: pd.DataFrame, threshold_2: float, threshold_3: float) -> pd.DataFrame:
    group_df = group_df.sort_values("XJS").copy()
    group_df["pred_two_stage"] = 0

    cand1 = group_df[group_df["is_candidate_1"] == 1]
    cand2 = group_df[group_df["is_candidate_2"] == 1]
    cand3 = group_df[group_df["is_candidate_3"] == 1]
    if cand1.empty:
        return group_df

    best_score = float("-inf")
    best = (None, None, None)

    for i in cand1.index:
        s1 = group_df.loc[i, "stage2_score_1"]
        # label 1 is mandatory
        if s1 > best_score:
            best_score = s1
            best = (i, None, None)

        cand2_valid = cand2[cand2.index >= i]
        for j in cand2_valid.index:
            s2 = group_df.loc[j, "stage2_score_2"]
            if s2 < threshold_2:
                continue
            total2 = s1 + s2
            if total2 > best_score:
                best_score = total2
                best = (i, j, None)

            cand3_valid = cand3[cand3.index >= j]
            for k in cand3_valid.index:
                s3 = group_df.loc[k, "stage2_score_3"]
                if s3 < threshold_3:
                    continue
                total3 = total2 + s3
                if total3 > best_score:
                    best_score = total3
                    best = (i, j, k)

    i, j, k = best
    if i is not None:
        group_df.loc[i, "pred_two_stage"] = 1
    if j is not None:
        group_df.loc[j, "pred_two_stage"] = 2
    if k is not None:
        group_df.loc[k, "pred_two_stage"] = 3
    return group_df


def search_thresholds(oof_df: pd.DataFrame) -> tuple[float, float, float]:
    best_score = -1.0
    best_pair = (0.3, 0.3)
    for t2 in np.arange(0.1, 0.91, 0.05):
        for t3 in np.arange(0.1, 0.91, 0.05):
            parts = []
            for _, group_df in oof_df.groupby(WELL_COL, sort=False):
                parts.append(decode_well(group_df, float(t2), float(t3)))
            pred_df = pd.concat(parts, ignore_index=True)
            score = f1_score(pred_df[LABEL_COL].astype(int), pred_df["pred_two_stage"].astype(int), average="macro")
            if score > best_score:
                best_score = score
                best_pair = (float(t2), float(t3))
    return best_pair[0], best_pair[1], best_score


def merge_candidate_scores(base_df: pd.DataFrame, cand_dfs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    out = base_df.copy()
    for label, cand_df in cand_dfs.items():
        temp = cand_df[["id", f"stage2_score_{label}", f"is_candidate_{label}"]].copy()
        out = out.merge(temp, on="id", how="left")
        out[f"is_candidate_{label}"] = out[f"is_candidate_{label}"].fillna(0).astype(int)
        out[f"stage2_score_{label}"] = out[f"stage2_score_{label}"].fillna(0.0)
    return out


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train_features, val_features, base_feature_cols = prepare_feature_frames()
    feat_cols = stage2_feature_columns(base_feature_cols)

    stage1_oof = load_stage1_oof()
    stage1_val = load_stage1_validation()
    train_df = add_stage1_features(train_features.merge(stage1_oof, on=["id", WELL_COL, "XJS", LABEL_COL], how="left"))
    val_df = add_stage1_features(val_features.merge(stage1_val, on=["id", WELL_COL, "XJS"], how="left"))

    oof_candidate_frames = {}
    val_candidate_frames = {}
    for label in [1, 2, 3]:
        train_cand = select_candidates(train_df, label)
        val_cand = select_candidates(val_df, label)
        train_cand = fit_label_oof(train_cand, label, feat_cols)
        val_cand = fit_label_full(train_cand, val_cand, label, feat_cols)
        oof_candidate_frames[label] = train_cand
        val_candidate_frames[label] = val_cand

    oof_full = merge_candidate_scores(train_df, oof_candidate_frames)
    val_full = merge_candidate_scores(val_df, val_candidate_frames)

    t2, t3, best_macro = search_thresholds(oof_full)

    oof_parts = [decode_well(group_df, t2, t3) for _, group_df in oof_full.groupby(WELL_COL, sort=False)]
    oof_pred = pd.concat(oof_parts, ignore_index=True).sort_values("id").reset_index(drop=True)
    val_parts = [decode_well(group_df, t2, t3) for _, group_df in val_full.groupby(WELL_COL, sort=False)]
    val_pred = pd.concat(val_parts, ignore_index=True).sort_values("id").reset_index(drop=True)

    metrics = {
        "model_name": "two_stage_candidates",
        "threshold_2": t2,
        "threshold_3": t3,
        "macro_f1_two_stage": f1_score(
            oof_pred[LABEL_COL].astype(int), oof_pred["pred_two_stage"].astype(int), average="macro"
        ),
        "macro_f1_stage1_structured_reference": 0.6413544839573021,
        "candidate_counts_per_well": TOP_K,
    }

    oof_pred.to_csv(REPORTS_DIR / "two_stage_oof_predictions.csv", index=False, encoding="utf-8-sig")
    val_pred.to_csv(REPORTS_DIR / "two_stage_validation_probabilities.csv", index=False, encoding="utf-8-sig")
    val_pred[["id", "pred_two_stage"]].rename(columns={"pred_two_stage": "关键点"}).to_csv(
        REPORTS_DIR / "two_stage_structured_predictions.csv", index=False, encoding="utf-8-sig"
    )
    with (REPORTS_DIR / "two_stage_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (REPORTS_DIR / "two_stage_summary.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Two-Stage Results\n\n"
            f"- threshold_2: `{t2:.2f}`\n"
            f"- threshold_3: `{t3:.2f}`\n"
            f"- two-stage macro-F1: `{metrics['macro_f1_two_stage']:.4f}`\n"
            f"- stage-1 structured reference: `{metrics['macro_f1_stage1_structured_reference']:.4f}`\n"
        )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

