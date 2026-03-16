from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


WELL_COL = "转换后JH"
LABEL_COL = "关键点"
ACTUAL_NUMERIC_COLS = ["XJS", "JX", "FW", "LJCZJS"]
DESIGN_NUMERIC_COLS = ["JX_design", "FW_design", "LJCZJS_design"]


@dataclass(frozen=True)
class DatasetBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    submission: pd.DataFrame


def dataset_dir(root: Path | str) -> Path:
    return Path(root) / "SINOPEC-02"


def load_bundle(root: Path | str) -> DatasetBundle:
    base = dataset_dir(root)
    return DatasetBundle(
        train=pd.read_csv(base / "train.csv"),
        validation=pd.read_csv(base / "validation_without_label.csv"),
        submission=pd.read_csv(base / "sample_submission.csv"),
    )


def split_actual_and_design(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if LABEL_COL in df.columns:
        actual_mask = df[LABEL_COL].notna()
    else:
        actual_mask = df[["JX", "FW"]].notna().all(axis=1)
    design_mask = df[DESIGN_NUMERIC_COLS].notna().any(axis=1)
    actual = df.loc[actual_mask].copy()
    design = df.loc[design_mask].copy()
    return actual, design


def sort_by_well_and_depth(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values([WELL_COL, "XJS", "id"]).reset_index(drop=True)


def circular_diff_deg(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diff = (a_arr - b_arr + 180.0) % 360.0 - 180.0
    return diff


def _interp_numeric(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x_src) & np.isfinite(y_src)
    if valid.sum() < 2:
        return np.full_like(x_dst, np.nan, dtype=float)
    return np.interp(x_dst, x_src[valid], y_src[valid], left=np.nan, right=np.nan)


def _interp_angle(x_src: np.ndarray, deg_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x_src) & np.isfinite(deg_src)
    if valid.sum() < 2:
        return np.full_like(x_dst, np.nan, dtype=float)
    rad = np.unwrap(np.deg2rad(deg_src[valid]))
    interp = np.interp(x_dst, x_src[valid], rad, left=np.nan, right=np.nan)
    return np.mod(np.rad2deg(interp), 360.0)


def attach_design_features(actual_df: pd.DataFrame, design_df: pd.DataFrame) -> pd.DataFrame:
    actual_df = sort_by_well_and_depth(actual_df)
    design_df = sort_by_well_and_depth(design_df)
    aligned_parts: list[pd.DataFrame] = []

    for well, g_actual in actual_df.groupby(WELL_COL, sort=False):
        g_actual = g_actual.copy()
        g_design = design_df.loc[design_df[WELL_COL] == well].sort_values("XJS")

        if g_design.empty:
            for col in DESIGN_NUMERIC_COLS:
                g_actual[f"{col}_aligned"] = np.nan
        else:
            x_actual = g_actual["XJS"].to_numpy(dtype=float)
            x_design = g_design["XJS"].to_numpy(dtype=float)
            g_actual["JX_design_aligned"] = _interp_numeric(
                x_design, g_design["JX_design"].to_numpy(dtype=float), x_actual
            )
            g_actual["LJCZJS_design_aligned"] = _interp_numeric(
                x_design, g_design["LJCZJS_design"].to_numpy(dtype=float), x_actual
            )
            g_actual["FW_design_aligned"] = _interp_angle(
                x_design, g_design["FW_design"].to_numpy(dtype=float), x_actual
            )

        g_actual["delta_JX_design"] = g_actual["JX"] - g_actual["JX_design_aligned"]
        g_actual["delta_LJCZJS_design"] = g_actual["LJCZJS"] - g_actual["LJCZJS_design_aligned"]
        g_actual["delta_FW_design"] = circular_diff_deg(g_actual["FW"], g_actual["FW_design_aligned"])
        aligned_parts.append(g_actual)

    return pd.concat(aligned_parts, ignore_index=True)


def per_well_counts(df: pd.DataFrame) -> pd.DataFrame:
    actual, design = split_actual_and_design(df)
    actual_counts = actual.groupby(WELL_COL).size().rename("actual_rows")
    design_counts = design.groupby(WELL_COL).size().rename("design_rows")
    return (
        pd.concat([actual_counts, design_counts], axis=1)
        .fillna(0)
        .astype(int)
        .reset_index()
        .sort_values(WELL_COL)
        .reset_index(drop=True)
    )


def unique_wells(df: pd.DataFrame) -> Iterable[str]:
    return df[WELL_COL].drop_duplicates().tolist()
