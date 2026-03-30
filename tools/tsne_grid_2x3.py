"""
t-SNE feature visualization: 3-row x 2-col grid  (two-stage pipeline).

Stage 1 — ``extract``
    Load models, balanced-sample **before** inference, extract features
    on 19LA/21LA/21DF eval sets, and save to a cache (.npz).

Stage 2 — ``plot``
    Load cache, run t-SNE, and produce a 3x2 PDF figure.
    No GPU or model weights needed — iterate on plot styling quickly.

Layout:
              ERM        IB-CAAN
    19LA    [ (a)          (b)   ]
    21LA    [ (c)          (d)   ]
    21DF    [ (e)          (f)   ]

Usage:
    # Stage 1 — feature extraction (needs GPU)
    python tools/tsne_grid_2x3.py extract \\
        --erm_config    config/Wav2vec2_XLSR_ASVspoof2019_BASELINE.conf \\
        --erm_ckpt      /path/to/erm.pth \\
        --ibcaan_config config/Wav2vec2_XLSR_ASVspoof2019_IBCAAN.conf \\
        --ibcaan_ckpt   /path/to/ibcaan.pth \\
        --cache         figures/tsne_cache.npz

    # Stage 2 — t-SNE + plot (CPU only, fast)
    python tools/tsne_grid_2x3.py plot \\
        --cache         figures/tsne_cache.npz \\
        --label_by class \\
        --output_pdf    figures/tsne_grid_class.pdf
"""

import argparse
import json
import random
import sys
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path so that data_utils / models / trainers
# can be imported regardless of how this script is invoked.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import (
    ATTACK_2_INT,
    Dataset_ASVspoof2021_eval,
)

# ── constants ────────────────────────────────────────────────────────
TRACKS = ["19LA", "21LA", "21DF"]
METHODS = ["ERM", "IB-CAAN"]
_CACHE_KEY = {"ERM": "ERM", "IB-CAAN": "IBCAAN"}
_CACHE_KEY_INV = {v: k for k, v in _CACHE_KEY.items()}

CLASS_COLORS = {1: "#1F77B4", 0: "#D62728"}      # blue / red
CLASS_NAMES  = {1: "Bonafide", 0: "Spoof"}

SUBPLOT_LABELS = [
    ["(a)", "(b)"],     # 19LA row
    ["(c)", "(d)"],     # 21LA row
    ["(e)", "(f)"],     # 21DF row
]


# =====================================================================
#  CLI
# =====================================================================
def parse_args() -> argparse.Namespace:
    root = argparse.ArgumentParser(
        description="3x2 t-SNE grid (two-stage: extract -> plot)."
    )
    sub = root.add_subparsers(dest="stage", required=True,
                              help="Pipeline stage.")

    # ── extract ──
    p_ext = sub.add_parser(
        "extract",
        help="Balanced-sample, extract features, save cache (.npz).")
    p_ext.add_argument("--erm_config",    type=str, required=True)
    p_ext.add_argument("--erm_ckpt",      type=str, required=True)
    p_ext.add_argument("--ibcaan_config", type=str, required=True)
    p_ext.add_argument("--ibcaan_ckpt",   type=str, required=True)
    p_ext.add_argument("--database_path", type=str, default=None,
                       help="Override database_path in configs.")
    p_ext.add_argument("--metadata_root", type=str,
                       default="eval-package/21/keys")
    p_ext.add_argument("--subset_21", type=str, default="eval",
                       choices=["eval", "progress", "hidden", "all"])
    p_ext.add_argument("--sample_by", type=str, default="class",
                       choices=["class", "attack"],
                       help="Label used for balanced sampling (default: class).")
    p_ext.add_argument("--max_samples", type=int, default=3000)
    p_ext.add_argument("--seed",        type=int, default=1234)
    p_ext.add_argument("--batch_size",  type=int, default=32)
    p_ext.add_argument("--num_workers", type=int, default=4)
    p_ext.add_argument("--cache", type=str, required=True,
                       help="Output cache path (.npz).")

    # ── plot ──
    p_plt = sub.add_parser(
        "plot",
        help="Load cache, run t-SNE, produce PDF.")
    p_plt.add_argument("--cache", type=str, required=True,
                       help="Input cache path (.npz).")
    p_plt.add_argument("--label_by", type=str, default="class",
                       choices=["class", "attack"],
                       help="Color labelling strategy.")
    p_plt.add_argument("--perplexity",  type=float, default=30.0)
    p_plt.add_argument("--n_iter",      type=int, default=1000)
    p_plt.add_argument("--seed",        type=int, default=1234)
    p_plt.add_argument("--output_pdf",  type=str,
                       default="figures/tsne_grid_3x2.pdf")

    return root.parse_args()


# =====================================================================
#  Helpers
# =====================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def normalize_device(config: Dict) -> str:
    return config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu"


# =====================================================================
#  Model / trainer
# =====================================================================
def build_model(model_config: Dict, device: torch.device):
    module = import_module(f"models.{model_config['architecture']}")
    return getattr(module, "Model")(model_config).to(device)


def build_trainer(model, device: torch.device, config: Dict):
    name = config["trainer"].lower()
    oc = dict(config["optim_config"])
    oc["epochs"] = config.get("num_epochs", 1)
    oc["steps_per_epoch"] = 1
    if name == "erm":
        from trainers.ERM_Trainer import ERM_Trainer
        return ERM_Trainer(model, device, oc, config)
    elif name == "ibcaan":
        from trainers.IBCAAN_Trainer import IBCAAN_Trainer
        t = IBCAAN_Trainer(model, device, oc, config)
        t.real_attack_id = ATTACK_2_INT["-"]
        return t
    raise ValueError(f"Unsupported trainer: {name}")


# =====================================================================
#  Metadata parsing
# =====================================================================
def parse_19_eval_metadata(config: Dict) -> List[Dict]:
    """Parse ASVspoof2019 LA eval protocol.

    Format: ``speaker utt_id - attack label``
    Attack codes: A07-A19 for spoof, "-" for bonafide.
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
            })
    if not records:
        raise ValueError("No records from 19LA eval protocol.")
    print(f"[Info] 19LA eval: {len(records)} total samples")
    return records


def parse_21_metadata(track: str, metadata_root: str, subset: str) -> List[Dict]:
    """Parse ASVspoof2021 trial_metadata.txt for LA or DF."""
    tk = "LA" if track == "21LA" else "DF"
    mf = Path(metadata_root) / tk / "CM" / "trial_metadata.txt"
    if not mf.exists():
        raise FileNotFoundError(f"Metadata not found: {mf}")
    records: List[Dict] = []
    with open(mf, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if not p:
                continue
            if track == "21LA":
                if len(p) < 8:
                    continue
                _, uid, _, _, atk, key, _, sub = p[:8]
            else:
                if len(p) < 9:
                    continue
                _, uid, _, _, atk, key, _, sub, _ = p[:9]
            if subset != "all" and sub != subset:
                continue
            records.append({"utt_id": uid,
                            "class": 1 if key == "bonafide" else 0,
                            "attack": atk})
    if not records:
        raise ValueError(f"No records for {track} subset={subset}")
    print(f"[Info] {track}: {len(records)} total samples (subset={subset})")
    return records


# =====================================================================
#  Balanced sampling  (on metadata, BEFORE building DataLoader)
# =====================================================================
def _balanced_sample_indices(
    labels: np.ndarray, max_n: int, seed: int
) -> np.ndarray:
    n = labels.shape[0]
    if max_n <= 0 or max_n >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    uniq = np.unique(labels)
    per_cls = max(1, max_n // max(1, len(uniq)))
    idx: List[int] = []
    for c in uniq:
        ci = np.where(labels == c)[0]
        idx.extend(rng.choice(ci, size=min(per_cls, len(ci)),
                              replace=False).tolist())
    if len(idx) < max_n:
        rest = np.setdiff1d(np.arange(n), np.array(idx))
        extra = min(max_n - len(idx), rest.shape[0])
        if extra > 0:
            idx.extend(rng.choice(rest, size=extra, replace=False).tolist())
    return np.array(sorted(idx), dtype=np.int64)


def sample_records(
    records: List[Dict], sample_by: str, max_n: int, seed: int
) -> List[Dict]:
    """Balanced-sample metadata records BEFORE building DataLoader."""
    if max_n <= 0 or max_n >= len(records):
        return records
    if sample_by == "class":
        labels = np.array([r["class"] for r in records], dtype=np.int64)
    else:
        labels = np.array([r["attack"] for r in records], dtype=str)
    keep = _balanced_sample_indices(labels, max_n, seed)
    return [records[int(i)] for i in keep]


# =====================================================================
#  Data loader
# =====================================================================
def get_audio_dir(config: Dict, track: str) -> Path:
    db = Path(config["database_path"])
    if track == "19LA":
        return db / "ASVspoof2019_LA" / "ASVspoof2019_LA_eval" / "flac"
    elif track == "21LA":
        return db / "ASVspoof2021_LA_eval" / "flac"
    else:
        return db / "ASVspoof2021_DF_eval" / "flac"


def build_loader(
    records: List[Dict], audio_dir: Path, bs: int, nw: int
) -> Tuple[DataLoader, Dict[str, Dict]]:
    """Build DataLoader from pre-sampled records."""
    fl = [r["utt_id"] for r in records]
    m2u = {r["utt_id"]: r for r in records}
    ds = Dataset_ASVspoof2021_eval(list_IDs=fl, base_dir=audio_dir)
    return DataLoader(ds, batch_size=bs, shuffle=False, drop_last=False,
                      pin_memory=torch.cuda.is_available(), num_workers=nw), m2u


# =====================================================================
#  Feature extraction
# =====================================================================
def extract_features(
    trainer, loader: DataLoader, meta: Dict[str, Dict], device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract latent features. Dataset returns ``(x, utt_id)``."""
    trainer.eval()
    feats, cls_l, atk_l = [], [], []
    with torch.no_grad():
        for bx, uids in loader:
            bx = bx.to(device, non_blocking=True)
            feats.append(trainer.get_features(bx).detach().cpu().numpy())
            for u in uids:
                m = meta[u]
                cls_l.append(m["class"])
                atk_l.append(m["attack"])
    return (np.concatenate(feats),
            np.array(cls_l, dtype=np.int64),
            np.array(atk_l, dtype=str))


# =====================================================================
#  Cache I/O
# =====================================================================
def save_cache(
    path: str,
    feat_dict: Dict[Tuple[str, str], np.ndarray],
    cls_dict:  Dict[Tuple[str, str], np.ndarray],
    atk_dict:  Dict[Tuple[str, str], np.ndarray],
    sample_by: str,
) -> None:
    out: Dict[str, np.ndarray] = {"_sample_by": np.array(sample_by)}
    for (method, track) in feat_dict:
        mk = _CACHE_KEY[method]
        out[f"{mk}_{track}_feat"] = feat_dict[(method, track)]
        out[f"{mk}_{track}_cls"]  = cls_dict[(method, track)]
        out[f"{mk}_{track}_atk"]  = atk_dict[(method, track)]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **out)
    size_mb = p.with_suffix(".npz").stat().st_size / 1024 / 1024
    print(f"[Done] Cache saved to: {p}  ({size_mb:.1f} MB)")


def load_cache(path: str) -> Tuple[
    Dict[Tuple[str, str], np.ndarray],
    Dict[Tuple[str, str], np.ndarray],
    Dict[Tuple[str, str], np.ndarray],
    str,
]:
    data = np.load(path, allow_pickle=False)
    sample_by = str(data["_sample_by"])
    feat_dict: Dict[Tuple[str, str], np.ndarray] = {}
    cls_dict:  Dict[Tuple[str, str], np.ndarray] = {}
    atk_dict:  Dict[Tuple[str, str], np.ndarray] = {}
    for mk, method in _CACHE_KEY_INV.items():
        for track in TRACKS:
            feat_dict[(method, track)] = data[f"{mk}_{track}_feat"]
            cls_dict[(method, track)]  = data[f"{mk}_{track}_cls"]
            atk_dict[(method, track)]  = data[f"{mk}_{track}_atk"]
    print(f"[Info] Cache loaded from: {path}  (sampled by: {sample_by})")
    for (method, track), f in feat_dict.items():
        print(f"       {method:8s} / {track}: "
              f"{f.shape[0]} samples, dim={f.shape[1]}")
    return feat_dict, cls_dict, atk_dict, sample_by


# =====================================================================
#  t-SNE
# =====================================================================
def run_tsne(features: np.ndarray, perplexity: float,
             n_iter: int, seed: int) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required. Install: pip install scikit-learn"
        ) from exc
    n = features.shape[0]
    perp = min(perplexity, max(5.0, (n - 1) / 3.0))
    if perp != perplexity:
        print(f"[Info] Perplexity adjusted: {perplexity} -> {perp:.1f} (n={n})")
    return TSNE(n_components=2, perplexity=perp, n_iter=n_iter,
                init="pca", learning_rate="auto",
                random_state=seed).fit_transform(features)


# =====================================================================
#  Plot helpers
# =====================================================================
def attack_display_name(value) -> str:
    """Human-readable attack label."""
    s = str(value)
    return "Bonafide" if s == "-" else s


def build_attack_colormap(
    track: str, labels_list: List[np.ndarray]
) -> Tuple[Dict, Dict]:
    """
    Build a unified color map for all unique attack values that appear in
    *any* of the cells belonging to the same track (row).
    """
    all_vals: set = set()
    for arr in labels_list:
        all_vals.update(arr.tolist())
    sorted_vals = sorted(all_vals, key=lambda x: str(x))

    cmap = plt.cm.get_cmap("tab20", max(len(sorted_vals), 1))
    color_map = {v: cmap(i) for i, v in enumerate(sorted_vals)}
    name_map  = {v: attack_display_name(v) for v in sorted_vals}
    return color_map, name_map


# =====================================================================
#  Plotting: 3 rows (tracks) x 2 cols (methods)
# =====================================================================
def plot_grid(
    embeddings:   Dict[Tuple[str, str], np.ndarray],
    label_arrays: Dict[Tuple[str, str], np.ndarray],
    label_by:     str,
    output_pdf:   str,
) -> None:
    plt.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size":        9,
        "axes.titlesize":   10,
        "axes.titleweight": "bold",
        "axes.labelsize":   8,
        "legend.fontsize":  7.5,
        "xtick.labelsize":  7,
        "ytick.labelsize":  7,
    })

    nrows, ncols = 3, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(5.8, 8.4))

    # ── pre-build per-row color maps for attack mode ──
    row_colormaps: Dict[str, Tuple[Dict, Dict]] = {}
    if label_by == "attack":
        for track in TRACKS:
            row_labels = [label_arrays[(m, track)] for m in METHODS]
            row_colormaps[track] = build_attack_colormap(track, row_labels)

    # ── scatter each cell ──
    for ri, track in enumerate(TRACKS):
        for ci, method in enumerate(METHODS):
            ax  = axes[ri][ci]
            key = (method, track)
            emb = embeddings[key]
            lbl = label_arrays[key]

            if label_by == "class":
                for v in sorted(np.unique(lbl)):
                    mask = lbl == v
                    ax.scatter(emb[mask, 0], emb[mask, 1],
                               s=4, alpha=0.55,
                               c=CLASS_COLORS.get(int(v), "#888"),
                               edgecolors="none",
                               label=CLASS_NAMES.get(int(v), str(v)),
                               rasterized=True)
            else:
                cmap_dict, nmap = row_colormaps[track]
                for v in sorted(cmap_dict.keys(), key=lambda x: str(x)):
                    mask = lbl == v
                    if not np.any(mask):
                        continue
                    ax.scatter(emb[mask, 0], emb[mask, 1],
                               s=4, alpha=0.55,
                               c=[cmap_dict[v]],
                               edgecolors="none",
                               label=nmap[v],
                               rasterized=True)

            # subplot label
            ax.text(0.03, 0.96, SUBPLOT_LABELS[ri][ci],
                    transform=ax.transAxes,
                    fontsize=9, fontweight="bold",
                    va="top", ha="left")

            # column title (top row only)
            if ri == 0:
                ax.set_title(method)

            # row label (left col only)
            if ci == 0:
                ax.set_ylabel(track, fontsize=10, fontweight="bold")

            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)

    # ── legends ──
    if label_by == "class":
        handles, labs = axes[0][0].get_legend_handles_labels()
        seen: dict = {}
        for h, l in zip(handles, labs):
            seen.setdefault(l, h)
        fig.legend(seen.values(), seen.keys(),
                   loc="lower center", ncol=len(seen),
                   frameon=False, bbox_to_anchor=(0.5, -0.01),
                   markerscale=2.5)
    else:
        for ri, track in enumerate(TRACKS):
            ax_right = axes[ri][1]
            handles, labs = ax_right.get_legend_handles_labels()
            seen = {}
            for h, l in zip(handles, labs):
                seen.setdefault(l, h)
            n_items = len(seen)
            legend = fig.legend(
                seen.values(), seen.keys(),
                loc="center",
                ncol=min(6, n_items),
                frameon=False,
                markerscale=2.0,
                bbox_to_anchor=(0.5, _row_legend_y(ri, nrows)),
                bbox_transform=fig.transFigure,
            )
            legend.set_in_layout(False)

    fig.subplots_adjust(
        left=0.08, right=0.97,
        top=0.95, bottom=0.05,
        wspace=0.12,
        hspace=0.35 if label_by == "attack" else 0.22,
    )

    out = Path(output_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Done] Saved 3x2 t-SNE grid to: {out}")


def _row_legend_y(row_idx: int, nrows: int) -> float:
    top, bot = 0.95, 0.05
    row_height = (top - bot) / nrows
    row_bottom = top - (row_idx + 1) * row_height
    return row_bottom - 0.015


# =====================================================================
#  Main — extract
# =====================================================================
def main_extract(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    erm_cfg    = read_config(args.erm_config)
    ibcaan_cfg = read_config(args.ibcaan_config)
    if args.database_path:
        erm_cfg["database_path"]    = args.database_path
        ibcaan_cfg["database_path"] = args.database_path
    if erm_cfg["database_path"] != ibcaan_cfg["database_path"]:
        raise ValueError("ERM and IB-CAAN must share the same database_path.")
    cfg = erm_cfg  # shared database_path

    device = torch.device(normalize_device(cfg))
    print(f"[Info] Device: {device}")

    # ── parse metadata ──
    all_records: Dict[str, List[Dict]] = {}
    all_records["19LA"] = parse_19_eval_metadata(cfg)
    all_records["21LA"] = parse_21_metadata("21LA", args.metadata_root,
                                            args.subset_21)
    all_records["21DF"] = parse_21_metadata("21DF", args.metadata_root,
                                            args.subset_21)

    # ── balanced sample BEFORE building DataLoader ──
    sampled: Dict[str, List[Dict]] = {}
    for track in TRACKS:
        sampled[track] = sample_records(
            all_records[track], args.sample_by, args.max_samples, args.seed)
        print(f"[Info] {track}: sampled {len(sampled[track])} / "
              f"{len(all_records[track])}")

    # ── build loaders from sampled records only ──
    loaders: Dict[str, Tuple] = {}
    for track in TRACKS:
        audio_dir = get_audio_dir(cfg, track)
        loaders[track] = build_loader(sampled[track], audio_dir,
                                      args.batch_size, args.num_workers)

    # ── build trainers ──
    erm_mcfg = dict(erm_cfg["model_config"])
    ib_mcfg  = dict(ibcaan_cfg["model_config"])
    erm_mcfg["device"] = str(device)
    ib_mcfg["device"]  = str(device)

    print("[Info] Building ERM model...")
    erm_trainer = build_trainer(build_model(erm_mcfg, device), device, erm_cfg)
    erm_trainer.load_checkpoint(args.erm_ckpt)
    erm_trainer.to(device)

    print("[Info] Building IB-CAAN model...")
    ib_trainer = build_trainer(build_model(ib_mcfg, device), device, ibcaan_cfg)
    ib_trainer.load_checkpoint(args.ibcaan_ckpt)
    ib_trainer.to(device)

    # ── extract features (only on sampled data) ──
    feat_dict: Dict[Tuple[str, str], np.ndarray] = {}
    cls_dict:  Dict[Tuple[str, str], np.ndarray] = {}
    atk_dict:  Dict[Tuple[str, str], np.ndarray] = {}

    for mname, trainer in [("ERM", erm_trainer), ("IB-CAAN", ib_trainer)]:
        for track in TRACKS:
            loader, meta = loaders[track]
            print(f"[Info] Extracting: {mname} / {track} ...")
            feat, cls, atk = extract_features(trainer, loader, meta, device)
            feat_dict[(mname, track)] = feat
            cls_dict[(mname, track)]  = cls
            atk_dict[(mname, track)]  = atk
            print(f"       -> {feat.shape[0]} samples, dim={feat.shape[1]}")

    # ── save cache ──
    save_cache(args.cache, feat_dict, cls_dict, atk_dict, args.sample_by)


# =====================================================================
#  Main — plot
# =====================================================================
def main_plot(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    feat_dict, cls_dict, atk_dict, sample_by = load_cache(args.cache)

    if args.label_by != sample_by:
        print(f"[Warn] Cache was sampled by '{sample_by}', "
              f"but plotting with label_by='{args.label_by}'. "
              f"Sampling balance may not match this label.")

    # ── t-SNE ──
    embeddings:   Dict[Tuple[str, str], np.ndarray] = {}
    label_arrays: Dict[Tuple[str, str], np.ndarray] = {}

    for method in METHODS:
        for track in TRACKS:
            key = (method, track)
            feat = feat_dict[key]
            lbl = cls_dict[key] if args.label_by == "class" else atk_dict[key]
            print(f"[Info] t-SNE: {method} / {track} "
                  f"({feat.shape[0]} samples) ...")
            embeddings[key]   = run_tsne(feat, args.perplexity,
                                         args.n_iter, args.seed)
            label_arrays[key] = lbl

    # ── plot ──
    plot_grid(embeddings, label_arrays, args.label_by, args.output_pdf)


# =====================================================================
#  Entry
# =====================================================================
def main() -> None:
    args = parse_args()
    if args.stage == "extract":
        main_extract(args)
    elif args.stage == "plot":
        main_plot(args)


if __name__ == "__main__":
    main()
