from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import LABEL_COL, WELL_COL
from .modeling import LABELS, decode_structured_predictions


class WellSequenceDataset(Dataset):
    def __init__(self, sequences: list[dict[str, torch.Tensor]]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.sequences[idx]


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    lengths = torch.tensor([item["x"].shape[0] for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    feat_dim = int(batch[0]["x"].shape[1])

    x = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    y = torch.full((len(batch), max_len), -100, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    meta = []
    for i, item in enumerate(batch):
        n = item["x"].shape[0]
        x[i, :n] = item["x"]
        y[i, :n] = item["y"]
        mask[i, :n] = True
        meta.append(item["meta"])

    return {"x": x, "y": y, "mask": mask, "lengths": lengths, "meta": meta}


class BiLSTMTagger(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(LABELS)),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return self.head(out)


@dataclass
class SequenceFoldResult:
    fold: int
    macro_f1_raw: float
    macro_f1_structured: float


def make_sequences(df: pd.DataFrame, feature_cols: list[str]) -> list[dict[str, torch.Tensor]]:
    sequences = []
    for well, g in df.groupby(WELL_COL, sort=False):
        g = g.sort_values("XJS").reset_index(drop=True)
        sequences.append(
            {
                "x": torch.tensor(g[feature_cols].to_numpy(dtype=np.float32)),
                "y": torch.tensor(g[LABEL_COL].astype(int).to_numpy(), dtype=torch.long),
                "meta": {
                    "well": well,
                    "id": g["id"].to_numpy(),
                    "xjs": g["XJS"].to_numpy(),
                },
            }
        )
    return sequences


def standardize_by_train(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1.0).fillna(1.0)
    train_df[feature_cols] = (train_df[feature_cols] - mean) / std
    valid_df[feature_cols] = (valid_df[feature_cols] - mean) / std
    train_df[feature_cols] = train_df[feature_cols].fillna(0.0)
    valid_df[feature_cols] = valid_df[feature_cols].fillna(0.0)
    return train_df, valid_df


def train_one_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    random_state: int,
    epochs: int = 18,
    batch_size: int = 8,
) -> pd.DataFrame:
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, valid_df = standardize_by_train(train_df, valid_df, feature_cols)
    train_sequences = make_sequences(train_df, feature_cols)
    valid_sequences = make_sequences(valid_df, feature_cols)

    train_loader = DataLoader(
        WellSequenceDataset(train_sequences), batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    valid_loader = DataLoader(
        WellSequenceDataset(valid_sequences), batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    model = BiLSTMTagger(input_dim=len(feature_cols)).to(device)

    labels = train_df[LABEL_COL].astype(int)
    counts = labels.value_counts().reindex(LABELS, fill_value=0).to_numpy(dtype=np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_state = None
    best_score = -1.0

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            lengths = batch["lengths"].to(device)
            logits = model(x, lengths)
            loss = criterion(logits.view(-1, logits.shape[-1]), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        preds = []
        truth = []
        with torch.no_grad():
            for batch in valid_loader:
                x = batch["x"].to(device)
                lengths = batch["lengths"].to(device)
                logits = model(x, lengths)
                prob = torch.softmax(logits, dim=-1).cpu()
                pred = prob.argmax(dim=-1)
                mask = batch["mask"]
                preds.extend(pred[mask].tolist())
                truth.extend(batch["y"][mask].tolist())
        score = f1_score(truth, preds, average="macro")
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    records = []
    with torch.no_grad():
        for batch in valid_loader:
            x = batch["x"].to(device)
            lengths = batch["lengths"].to(device)
            logits = model(x, lengths)
            prob = torch.softmax(logits, dim=-1).cpu()
            for i, meta in enumerate(batch["meta"]):
                n = len(meta["id"])
                row = pd.DataFrame(
                    {
                        "id": meta["id"],
                        WELL_COL: meta["well"],
                        "XJS": meta["xjs"],
                        LABEL_COL: batch["y"][i, :n].cpu().numpy(),
                    }
                )
                row["pred_raw"] = prob[i, :n].argmax(dim=-1).numpy()
                for label in LABELS:
                    row[f"prob_{label}"] = prob[i, :n, label].numpy()
                records.append(row)

    out = pd.concat(records, ignore_index=True)
    structured_parts = []
    prob_cols = [f"prob_{label}" for label in LABELS]
    for _, group_df in out.groupby(WELL_COL, sort=False):
        ordered = group_df.sort_values("XJS").copy()
        ordered["pred_structured"] = decode_structured_predictions(ordered, prob_cols)
        structured_parts.append(ordered)
    return pd.concat(structured_parts, ignore_index=True)


def cross_validate_sequence(
    data: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path | str,
    n_splits: int = 5,
    output_prefix: str = "bilstm",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = data[WELL_COL]
    y = data[LABEL_COL].astype(int)
    splitter = GroupKFold(n_splits=n_splits)

    oof_parts = []
    fold_results = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(data[feature_cols], y, groups), start=1):
        train_df = data.iloc[train_idx].copy()
        valid_df = data.iloc[valid_idx].copy()
        fold_pred = train_one_fold(train_df, valid_df, feature_cols, random_state=100 + fold)
        oof_parts.append(fold_pred)
        fold_results.append(
            SequenceFoldResult(
                fold=fold,
                macro_f1_raw=f1_score(fold_pred[LABEL_COL], fold_pred["pred_raw"], average="macro"),
                macro_f1_structured=f1_score(
                    fold_pred[LABEL_COL], fold_pred["pred_structured"], average="macro"
                ),
            )
        )

    oof = pd.concat(oof_parts, ignore_index=True).sort_values("id").reset_index(drop=True)
    metrics = {
        "model_name": "bilstm_sequence",
        "fold_results": [r.__dict__ for r in fold_results],
        "overall": {
            "macro_f1_raw": f1_score(oof[LABEL_COL], oof["pred_raw"], average="macro"),
            "macro_f1_structured": f1_score(
                oof[LABEL_COL], oof["pred_structured"], average="macro"
            ),
        },
    }

    oof.to_csv(output_dir / f"{output_prefix}_oof_predictions.csv", index=False, encoding="utf-8-sig")
    with (output_dir / f"{output_prefix}_cv_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics

