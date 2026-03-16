from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LABEL_COL, WELL_COL, circular_diff_deg


ROLLING_WINDOWS = (3, 5, 9)


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def build_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    parts: list[pd.DataFrame] = []

    for well, g in df.groupby(WELL_COL, sort=False):
        g = g.sort_values("XJS").reset_index(drop=True).copy()
        dx = g["XJS"].diff()

        g["well_index"] = np.arange(len(g))
        g["well_length"] = len(g)
        g["well_pos_frac"] = np.linspace(0.0, 1.0, len(g))
        g["FW_sin"] = np.sin(np.deg2rad(g["FW"]))
        g["FW_cos"] = np.cos(np.deg2rad(g["FW"]))

        g["dXJS"] = dx
        g["dJX"] = _safe_divide(g["JX"].diff(), dx)
        g["d2JX"] = _safe_divide(g["dJX"].diff(), dx)
        fw_step = pd.Series(circular_diff_deg(g["FW"], g["FW"].shift(1)), index=g.index)
        g["dFW"] = _safe_divide(fw_step, dx)
        g["abs_dFW"] = g["dFW"].abs()
        g["dLJCZJS"] = _safe_divide(g["LJCZJS"].diff(), dx)
        g["JX_minus_prev"] = g["JX"].diff()
        g["JX_minus_next"] = g["JX"] - g["JX"].shift(-1)

        for window in ROLLING_WINDOWS:
            centered = g["JX"].rolling(window=window, center=True, min_periods=1)
            g[f"JX_roll_mean_{window}"] = centered.mean()
            g[f"JX_roll_std_{window}"] = centered.std().fillna(0.0)
            slope_centered = g["dJX"].rolling(window=window, center=True, min_periods=1)
            g[f"dJX_roll_mean_{window}"] = slope_centered.mean()
            g[f"dJX_roll_std_{window}"] = slope_centered.std().fillna(0.0)

        if "JX_design_aligned" in g.columns:
            g["has_design_alignment"] = g["JX_design_aligned"].notna().astype(int)
            g["delta_JX_design_abs"] = g["delta_JX_design"].abs()
            g["delta_FW_design_abs"] = g["delta_FW_design"].abs()
        else:
            g["has_design_alignment"] = 0

        parts.append(g)

    features = pd.concat(parts, ignore_index=True)

    feature_cols = [
        col
        for col in features.columns
        if col
        not in {
            "id",
            WELL_COL,
            LABEL_COL,
            "JX_design",
            "FW_design",
            "LJCZJS_design",
        }
        and features[col].dtype != "O"
    ]

    features[feature_cols] = features[feature_cols].replace([np.inf, -np.inf], np.nan)
    return features, feature_cols
