import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from analysis_common import safe_attack_name, write_rows_csv


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Train linear probes on exported frozen features.")
    parser.add_argument("--feature-files", nargs="+", required=True, help="Paths to .pt files exported by export_embeddings.py")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory to store probe summaries.")
    parser.add_argument("--attack-field", default=None, type=str, help="Metadata field used as spoof-type label. Defaults to attack_id.")
    parser.add_argument("--train-ratio", default=0.7, type=float, help="Train split ratio for probe fitting.")
    parser.add_argument("--epochs", default=100, type=int, help="Training epochs for each probe.")
    parser.add_argument("--lr", default=1e-2, type=float, help="Learning rate for probe fitting.")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="Weight decay for probe fitting.")
    parser.add_argument("--runs", default=3, type=int, help="Repeated random splits.")
    parser.add_argument("--seed", default=1234, type=int, help="Base random seed.")
    parser.add_argument("--device", default="cuda", type=str, help="Device for probe fitting.")
    return parser.parse_args()


def stratified_split(labels: np.ndarray, train_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_train = max(1, int(len(cls_idx) * train_ratio))
        if n_train >= len(cls_idx):
            n_train = len(cls_idx) - 1
        train_idx.extend(cls_idx[:n_train].tolist())
        test_idx.extend(cls_idx[n_train:].tolist())
    return np.array(train_idx), np.array(test_idx)


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def fit_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> float:
    train_x_t = torch.tensor(train_x, dtype=torch.float32, device=device)
    test_x_t = torch.tensor(test_x, dtype=torch.float32, device=device)
    train_y_t = torch.tensor(train_y, dtype=torch.long, device=device)
    test_y_t = torch.tensor(test_y, dtype=torch.long, device=device)

    probe = LinearProbe(train_x.shape[1], int(train_y.max()) + 1).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad()
        logits = probe(train_x_t)
        loss = criterion(logits, train_y_t)
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(test_x_t).argmax(dim=1)
    acc = (pred == test_y_t).float().mean().item()
    return acc


def encode_attack_labels(rows: List[Dict[str, object]], attack_field: Optional[str]) -> np.ndarray:
    field = attack_field or "attack_id"
    values: List[Optional[str]] = []
    for row in rows:
        value = safe_attack_name(row, field)
        if value is None and field != "attack_id":
            value = safe_attack_name(row, "attack_id")
        values.append(value)
    unique = sorted({v for v in values if v is not None})
    mapping = {name: idx for idx, name in enumerate(unique)}
    return np.array([mapping.get(v, -1) for v in values], dtype=np.int64)


def evaluate_feature_file(feature_file: Path, args) -> Dict[str, object]:
    payload = torch.load(feature_file, map_location="cpu")
    features = np.asarray(payload["features"])
    rows = payload["rows"]
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    attack_labels = encode_attack_labels(rows, args.attack_field)

    binary_scores: List[float] = []
    spoof_scores: List[float] = []

    for run_id in range(args.runs):
        seed = args.seed + run_id
        train_idx, test_idx = stratified_split(labels, args.train_ratio, seed)
        train_x, test_x = standardize(features[train_idx], features[test_idx])
        binary_acc = fit_probe(
            train_x, labels[train_idx], test_x, labels[test_idx],
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, device=args.device
        )
        binary_scores.append(binary_acc)

        spoof_mask = labels == 0
        spoof_indices = np.where(spoof_mask & (attack_labels >= 0))[0]
        if spoof_indices.size > 0 and np.unique(attack_labels[spoof_indices]).size >= 2:
            spoof_train_rel, spoof_test_rel = stratified_split(
                attack_labels[spoof_indices], args.train_ratio, seed
            )
            spoof_train_idx = spoof_indices[spoof_train_rel]
            spoof_test_idx = spoof_indices[spoof_test_rel]
            s_train_x, s_test_x = standardize(features[spoof_train_idx], features[spoof_test_idx])
            spoof_acc = fit_probe(
                s_train_x, attack_labels[spoof_train_idx], s_test_x, attack_labels[spoof_test_idx],
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, device=args.device
            )
            spoof_scores.append(spoof_acc)

    binary_mean = float(np.mean(binary_scores))
    binary_std = float(np.std(binary_scores))
    spoof_mean = float(np.mean(spoof_scores)) if spoof_scores else math.nan
    spoof_std = float(np.std(spoof_scores)) if spoof_scores else math.nan

    result = {
        "feature_file": str(feature_file),
        "feature_source": rows[0].get("feature_source", feature_file.stem) if rows else feature_file.stem,
        "num_samples": len(rows),
        "binary_acc_mean": binary_mean,
        "binary_acc_std": binary_std,
        "spoof_type_acc_mean": spoof_mean,
        "spoof_type_acc_std": spoof_std,
        "relative_gap": binary_mean - spoof_mean if not math.isnan(spoof_mean) else math.nan,
        "attack_field": args.attack_field or "attack_id",
    }
    return result


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for feature_file in args.feature_files:
        result = evaluate_feature_file(Path(feature_file), args)
        results.append(result)

    write_rows_csv(output_dir / "probe_summary.csv", results)
    with open(output_dir / "probe_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved probe results to {output_dir}")


if __name__ == "__main__":
    main()
