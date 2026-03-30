"""
t-SNE feature visualization: ERM vs IB-CAAN (single track, 1x2 panel).

All tracks (19LA / 21LA / 21DF) use the **eval** set.

Usage example:
    python tools/tsne_compare_erm_ibcaan.py \\
        --erm_config    config/Wav2vec2_XLSR_ASVspoof2019_BASELINE.conf \\
        --erm_ckpt      /path/to/erm.pth \\
        --ibcaan_config config/Wav2vec2_XLSR_ASVspoof2019_IBCAAN.conf \\
        --ibcaan_ckpt   /path/to/ibcaan.pth \\
        --track 19LA \\
        --label_by class \\
        --output_pdf tsne_erm_vs_ibcaan.pdf
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path so that data_utils / models / trainers
# can be imported regardless of how this script is invoked.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import (
    ATTACK_2_INT,
    Dataset_ASVspoof2021_eval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="t-SNE feature visualization for ERM vs IB-CAAN."
    )
    parser.add_argument(
        "--erm_config",
        type=str,
        required=True,
        help="Path to ERM config (.conf / json).",
    )
    parser.add_argument(
        "--erm_ckpt",
        type=str,
        required=True,
        help="Path to ERM checkpoint (.pth).",
    )
    parser.add_argument(
        "--ibcaan_config",
        type=str,
        required=True,
        help="Path to IB-CAAN config (.conf / json).",
    )
    parser.add_argument(
        "--ibcaan_ckpt",
        type=str,
        required=True,
        help="Path to IB-CAAN checkpoint (.pth).",
    )
    parser.add_argument(
        "--database_path",
        type=str,
        default=None,
        help="Override database_path in config.",
    )
    parser.add_argument(
        "--track",
        type=str,
        default="19LA",
        choices=["19LA", "21LA", "21DF"],
        help="Dataset track for visualization.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="eval",
        choices=["eval", "progress", "hidden", "all"],
        help="Subset filter for 21LA/21DF metadata (default: eval).",
    )
    parser.add_argument(
        "--metadata_root",
        type=str,
        default="eval-package/21/keys",
        help="Root of ASVspoof2021 key metadata, e.g., eval-package/21/keys.",
    )
    parser.add_argument(
        "--label_by",
        type=str,
        default="class",
        choices=["class", "attack", "codec", "transmission", "source", "vocoder"],
        help="Color labels in t-SNE plot.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=3000,
        help="Max samples for t-SNE (applied after loading all data).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity.",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1000,
        help="t-SNE optimization iterations.",
    )
    parser.add_argument(
        "--output_pdf",
        type=str,
        default="tsne_erm_vs_ibcaan.pdf",
        help="Output PDF path.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def build_model_from_config(model_config: Dict, device: torch.device):
    from importlib import import_module

    module = import_module(f"models.{model_config['architecture']}")
    model_cls = getattr(module, "Model")
    model = model_cls(model_config).to(device)
    return model


def build_trainer(model, device: torch.device, config: Dict):
    trainer_name = config["trainer"].lower()
    optim_config = dict(config["optim_config"])
    optim_config["epochs"] = config.get("num_epochs", 1)
    optim_config["steps_per_epoch"] = 1

    if trainer_name == "erm":
        from trainers.ERM_Trainer import ERM_Trainer

        trainer = ERM_Trainer(model, device, optim_config, config)
    elif trainer_name == "ibcaan":
        from trainers.IBCAAN_Trainer import IBCAAN_Trainer

        trainer = IBCAAN_Trainer(model, device, optim_config, config)
        trainer.real_attack_id = ATTACK_2_INT["-"]
    else:
        raise ValueError(f"Unsupported trainer for visualization: {trainer_name}")
    return trainer


def validate_label_by(track: str, label_by: str) -> None:
    """Check that *label_by* is valid for the given track."""
    valid = {
        "19LA": {"class", "attack"},
        "21LA": {"class", "attack", "codec", "transmission"},
        "21DF": {"class", "attack", "codec", "source", "vocoder"},
    }
    if label_by not in valid[track]:
        raise ValueError(
            f"label_by={label_by} is not supported for track={track}. "
            f"Valid choices: {sorted(valid[track])}"
        )


# =====================================================================
#  Metadata parsing
# =====================================================================
def parse_19_eval_metadata(config: Dict) -> List[Dict]:
    """Parse ASVspoof2019 LA eval protocol.

    Format: ``speaker utt_id - attack label``
    Attack codes are A07-A19 for spoof, "-" for bonafide.
    """
    db = Path(config["database_path"])
    meta_file = (db / "ASVspoof2019_LA"
                 / "ASVspoof2019_LA_cm_protocols"
                 / "ASVspoof2019.LA.cm.eval.trl.txt")
    if not meta_file.exists():
        raise FileNotFoundError(f"19LA eval protocol not found: {meta_file}")
    records: List[Dict] = []
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, utt_id, _, attack, label = parts[:5]
            records.append({
                "utt_id": utt_id,
                "class": 1 if label == "bonafide" else 0,
                "attack": attack,
                "codec": "N/A",
                "transmission": "N/A",
                "source": "N/A",
                "vocoder": "N/A",
            })
    if not records:
        raise ValueError("No records parsed from 19LA eval protocol.")
    print(f"[Info] 19LA eval: {len(records)} samples")
    return records


def parse_21_metadata(track: str, metadata_root: str, subset: str) -> List[Dict]:
    """Parse ASVspoof2021 trial_metadata.txt for LA or DF."""
    track_name = "LA" if track == "21LA" else "DF"
    meta_file = Path(metadata_root) / track_name / "CM" / "trial_metadata.txt"
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")

    records: List[Dict] = []
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if track == "21LA":
                if len(parts) < 8:
                    continue
                spk, utt_id, codec, transmission, attack, key, trim, sub = parts[:8]
                if subset != "all" and sub != subset:
                    continue
                records.append(
                    {
                        "utt_id": utt_id,
                        "class": 1 if key == "bonafide" else 0,
                        "attack": attack,
                        "codec": codec,
                        "transmission": transmission,
                        "source": "N/A",
                        "vocoder": "N/A",
                    }
                )
            else:
                if len(parts) < 9:
                    continue
                spk, utt_id, codec, source, attack, key, trim, sub, vocoder = parts[:9]
                if subset != "all" and sub != subset:
                    continue
                records.append(
                    {
                        "utt_id": utt_id,
                        "class": 1 if key == "bonafide" else 0,
                        "attack": attack,
                        "codec": codec,
                        "transmission": "N/A",
                        "source": source,
                        "vocoder": vocoder,
                    }
                )
    if len(records) == 0:
        raise ValueError(
            f"No records found in {meta_file} after subset filter: subset={subset}"
        )
    print(f"[Info] Parsed {len(records)} metadata records from: {meta_file}")
    return records


# =====================================================================
#  Data loaders  (all tracks use eval set + Dataset_ASVspoof2021_eval)
# =====================================================================
def build_loader_19(
    config: Dict, batch_size: int, num_workers: int
) -> Tuple[DataLoader, Dict[str, Dict]]:
    """Build 19LA eval loader. Returns (DataLoader, meta_by_utt)."""
    database_path = Path(config["database_path"])
    audio_dir = (database_path / "ASVspoof2019_LA"
                 / "ASVspoof2019_LA_eval" / "flac")
    records = parse_19_eval_metadata(config)
    file_list = [r["utt_id"] for r in records]
    meta_by_utt = {r["utt_id"]: r for r in records}
    dataset = Dataset_ASVspoof2021_eval(list_IDs=file_list, base_dir=audio_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )
    return loader, meta_by_utt


def build_loader_21(
    config: Dict,
    track: str,
    metadata_root: str,
    subset: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, Dict[str, Dict]]:
    """Build 21LA / 21DF eval loader. Returns (DataLoader, meta_by_utt)."""
    database_path = Path(config["database_path"])
    if track == "21LA":
        eval_database_path = database_path / "ASVspoof2021_LA_eval" / "flac"
    else:
        eval_database_path = database_path / "ASVspoof2021_DF_eval" / "flac"

    records = parse_21_metadata(track, metadata_root, subset)
    file_list = [r["utt_id"] for r in records]
    meta_by_utt = {r["utt_id"]: r for r in records}

    dataset = Dataset_ASVspoof2021_eval(list_IDs=file_list, base_dir=eval_database_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )
    return loader, meta_by_utt


# =====================================================================
#  Feature extraction  (unified – all tracks use Dataset_ASVspoof2021_eval)
# =====================================================================
def extract_features(
    trainer,
    loader: DataLoader,
    meta_by_utt: Dict[str, Dict],
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Extract latent features. Dataset returns ``(x, utt_id)``."""
    trainer.eval()
    feat_list: List[np.ndarray] = []
    class_list: List[int] = []
    attack_list: List[str] = []
    codec_list: List[str] = []
    transmission_list: List[str] = []
    source_list: List[str] = []
    vocoder_list: List[str] = []

    with torch.no_grad():
        for batch_x, utt_ids in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            z = trainer.get_features(batch_x).detach().cpu().numpy()
            feat_list.append(z)

            for uid in utt_ids:
                meta = meta_by_utt[uid]
                class_list.append(meta["class"])
                attack_list.append(meta["attack"])
                codec_list.append(meta["codec"])
                transmission_list.append(meta["transmission"])
                source_list.append(meta["source"])
                vocoder_list.append(meta["vocoder"])

    feats = np.concatenate(feat_list, axis=0)
    labels = {
        "class": np.array(class_list, dtype=np.int64),
        "attack": np.array(attack_list, dtype=object),
        "codec": np.array(codec_list, dtype=object),
        "transmission": np.array(transmission_list, dtype=object),
        "source": np.array(source_list, dtype=object),
        "vocoder": np.array(vocoder_list, dtype=object),
    }
    return feats, labels


def sample_indices(labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    n = labels.shape[0]
    if max_samples <= 0 or max_samples >= n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    unique = np.unique(labels)
    indices: List[int] = []
    per_cls = max(1, max_samples // max(1, len(unique)))
    for cls in unique:
        cls_idx = np.where(labels == cls)[0]
        take = min(per_cls, len(cls_idx))
        picked = rng.choice(cls_idx, size=take, replace=False)
        indices.extend(picked.tolist())

    if len(indices) < max_samples:
        remain = np.setdiff1d(np.arange(n), np.array(indices, dtype=np.int64))
        extra = min(max_samples - len(indices), remain.shape[0])
        if extra > 0:
            more = rng.choice(remain, size=extra, replace=False)
            indices.extend(more.tolist())

    return np.array(sorted(indices), dtype=np.int64)


def run_tsne(features: np.ndarray, perplexity: float, n_iter: int, seed: int) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for t-SNE. Install with: pip install scikit-learn"
        ) from exc

    n_samples = features.shape[0]
    max_valid_perplexity = max(5.0, float(n_samples - 1) / 3.0)
    used_perplexity = min(perplexity, max_valid_perplexity)
    if used_perplexity != perplexity:
        print(
            f"[Info] Perplexity auto-adjusted from {perplexity} to {used_perplexity:.2f} "
            f"for n_samples={n_samples}."
        )

    tsne = TSNE(
        n_components=2,
        perplexity=used_perplexity,
        n_iter=n_iter,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(features)


def value_to_display(label_by: str, value) -> str:
    """Human-readable display string for a label value."""
    if label_by == "class":
        return "Bonafide" if int(value) == 1 else "Spoof"
    if label_by == "attack":
        s = str(value)
        return "Bonafide" if s == "-" else s
    return str(value)


def get_colors(unique_values: np.ndarray, label_by: str) -> Dict:
    if label_by == "class":
        color_map = {
            1: (0.12, 0.47, 0.71, 0.8),  # blue
            0: (0.84, 0.15, 0.16, 0.8),  # red
        }
        return {v: color_map.get(int(v), (0.3, 0.3, 0.3, 0.8)) for v in unique_values}

    cmap = plt.cm.get_cmap("tab20", len(unique_values))
    return {v: cmap(i) for i, v in enumerate(unique_values)}


def plot_two_panel_pdf(
    emb_erm: np.ndarray,
    emb_ib: np.ndarray,
    labels: np.ndarray,
    label_by: str,
    output_pdf: str,
) -> None:
    unique_values = np.unique(labels)
    colors = get_colors(unique_values, label_by)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    panels = [("ERM", emb_erm, axes[0]), ("IB-CAAN", emb_ib, axes[1])]

    for title, emb, ax in panels:
        for v in unique_values:
            mask = labels == v
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=9,
                alpha=0.7,
                c=[colors[v]],
                edgecolors="none",
                label=value_to_display(label_by, v),
            )

        ax.set_title(title)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.45)

    handles, legend_labels = axes[1].get_legend_handles_labels()
    unique_pairs = {}
    for h, l in zip(handles, legend_labels):
        if l not in unique_pairs:
            unique_pairs[l] = h
    fig.legend(
        unique_pairs.values(),
        unique_pairs.keys(),
        loc="lower center",
        ncol=min(6, len(unique_pairs)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    output_path = Path(output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] t-SNE PDF saved to: {output_path}")


def normalize_device(config: Dict) -> str:
    if torch.cuda.is_available():
        return config.get("device", "cuda:0")
    return "cpu"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    validate_label_by(args.track, args.label_by)

    erm_config = read_config(args.erm_config)
    ibcaan_config = read_config(args.ibcaan_config)

    if args.database_path is not None:
        erm_config["database_path"] = args.database_path
        ibcaan_config["database_path"] = args.database_path

    if erm_config["database_path"] != ibcaan_config["database_path"]:
        raise ValueError("ERM and IB-CAAN configs must use the same database_path.")

    device = torch.device(normalize_device(erm_config))
    print(f"[Info] Using device: {device}")
    print(
        f"[Info] track={args.track}, subset={args.subset}, "
        f"label_by={args.label_by}"
    )

    erm_model_config = dict(erm_config["model_config"])
    ib_model_config = dict(ibcaan_config["model_config"])
    erm_model_config["device"] = str(device)
    ib_model_config["device"] = str(device)

    # ── data loader (all tracks use eval set) ──
    if args.track == "19LA":
        loader, meta_by_utt = build_loader_19(
            erm_config, args.batch_size, args.num_workers)
    else:
        loader, meta_by_utt = build_loader_21(
            erm_config,
            args.track,
            args.metadata_root,
            args.subset,
            args.batch_size,
            args.num_workers,
        )

    erm_model = build_model_from_config(erm_model_config, device)
    erm_trainer = build_trainer(erm_model, device, erm_config)
    erm_trainer.load_checkpoint(args.erm_ckpt)
    print(f"[Info] Loaded ERM checkpoint: {args.erm_ckpt}")

    ib_model = build_model_from_config(ib_model_config, device)
    ib_trainer = build_trainer(ib_model, device, ibcaan_config)
    ib_trainer.load_checkpoint(args.ibcaan_ckpt)
    print(f"[Info] Loaded IB-CAAN checkpoint: {args.ibcaan_ckpt}")

    print("[Info] Extracting ERM features...")
    feat_erm, labels_erm = extract_features(erm_trainer, loader, meta_by_utt, device)

    print("[Info] Extracting IB-CAAN features...")
    feat_ib, labels_ib = extract_features(ib_trainer, loader, meta_by_utt, device)

    for k in labels_erm.keys():
        if not np.array_equal(labels_erm[k], labels_ib[k]):
            raise RuntimeError(f"Label mismatch between ERM and IB-CAAN for key={k}.")

    labels = labels_erm[args.label_by]
    keep_idx = sample_indices(labels, args.max_samples, args.seed)
    feat_erm = feat_erm[keep_idx]
    feat_ib = feat_ib[keep_idx]
    labels = labels[keep_idx]

    print(f"[Info] Running t-SNE on {len(keep_idx)} samples per model...")
    emb_erm = run_tsne(feat_erm, args.perplexity, args.n_iter, args.seed)
    emb_ib = run_tsne(feat_ib, args.perplexity, args.n_iter, args.seed)

    plot_two_panel_pdf(
        emb_erm=emb_erm,
        emb_ib=emb_ib,
        labels=labels,
        label_by=args.label_by,
        output_pdf=args.output_pdf,
    )


if __name__ == "__main__":
    main()
