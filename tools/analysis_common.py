import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_utils import ATTACK_2_INT, pad
from main import get_model, get_trainer


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SampleRecord:
    utt_id: str
    audio_path: Path
    label: int
    label_name: str
    attack_id: str
    attack_group: int
    track: str
    split: str
    metadata: Dict[str, str]


class AudioProtocolDataset(Dataset):
    def __init__(self, samples: List[SampleRecord], cut: int = 64600, sr: int = 16000):
        self.samples = samples
        self.cut = cut
        self.sr = sr

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        waveform, _ = librosa.load(sample.audio_path, sr=self.sr)
        padded = pad(waveform, self.cut)
        return {
            "audio": torch.tensor(padded, dtype=torch.float32),
            "utt_id": sample.utt_id,
            "label": sample.label,
            "label_name": sample.label_name,
            "attack_id": sample.attack_id,
            "attack_group": sample.attack_group,
            "track": sample.track,
            "split": sample.split,
            "metadata": sample.metadata,
        }


def collate_analysis_batch(batch: List[Dict]) -> Dict[str, object]:
    audio = torch.stack([item["audio"] for item in batch], dim=0)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    attack_groups = torch.tensor([item["attack_group"] for item in batch], dtype=torch.long)
    return {
        "audio": audio,
        "labels": labels,
        "utt_ids": [item["utt_id"] for item in batch],
        "label_names": [item["label_name"] for item in batch],
        "attack_ids": [item["attack_id"] for item in batch],
        "attack_groups": attack_groups,
        "tracks": [item["track"] for item in batch],
        "splits": [item["split"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["model_config"]["device"] = config["device"]
    config["optim_config"]["epochs"] = config["num_epochs"]
    return config


def build_trainer_from_config(config_path: str):
    config = load_config(config_path)
    device = config["device"] if torch.cuda.is_available() else "cpu"
    model = get_model(config["model_config"], device)
    trainer = get_trainer(model, device, config["optim_config"], config)
    trainer.to(device)
    trainer.eval()
    return config, trainer, device


def resolve_checkpoint_paths(config: Dict) -> Tuple[List[str], str]:
    model_dir = Path(config["model_path"])
    checkpoints = [str(model_dir / ckpt) for ckpt in config["checkpoints"]]
    average_path = str(model_dir / "model_avg.pth")
    return checkpoints, average_path


def load_model_weights(trainer, config: Dict) -> str:
    checkpoints, average_path = resolve_checkpoint_paths(config)
    if len(checkpoints) == 1:
        trainer.load_checkpoint(checkpoints[0])
        return checkpoints[0]
    trainer.load_average_checkpoint(checkpoints, average_path)
    return average_path


def build_analysis_loader(
    config_path: str,
    database_path: str,
    track: str,
    split: str = "eval",
    subset: str = "all",
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
):
    config = load_config(config_path)
    db_root = Path(database_path)
    samples = build_samples(db_root, track=track, split=split, subset=subset)
    dataset = AudioProtocolDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size or config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=config["num_workers"] if num_workers is None else num_workers,
        collate_fn=collate_analysis_batch,
    )
    return config, samples, loader


def build_samples(database_path: Path, track: str, split: str, subset: str) -> List[SampleRecord]:
    track = track.upper()
    split = split.lower()
    subset = subset.lower()

    if track == "19LA":
        return _build_2019_samples(database_path, split)
    if track == "21LA":
        return _build_2021_la_samples(database_path, subset)
    if track == "21DF":
        return _build_2021_df_samples(database_path, subset)
    if track == "ITW":
        return _build_itw_samples(database_path)
    raise ValueError(f"Unsupported track: {track}")


def _build_2019_samples(database_path: Path, split: str) -> List[SampleRecord]:
    protocol_map = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }
    audio_dir_map = {
        "train": database_path / "ASVspoof2019_LA" / "ASVspoof2019_LA_train" / "flac",
        "dev": database_path / "ASVspoof2019_LA" / "ASVspoof2019_LA_dev" / "flac",
        "eval": database_path / "ASVspoof2019_LA" / "ASVspoof2019_LA_eval" / "flac",
    }
    protocol_path = database_path / "ASVspoof2019_LA" / "ASVspoof2019_LA_cm_protocols" / protocol_map[split]
    audio_dir = audio_dir_map[split]

    samples: List[SampleRecord] = []
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            speaker, utt_id, _, attack_id, label_name = line.strip().split()
            label = 1 if label_name == "bonafide" else 0
            attack_group = ATTACK_2_INT.get(attack_id, -1)
            samples.append(
                SampleRecord(
                    utt_id=utt_id,
                    audio_path=audio_dir / f"{utt_id}.flac",
                    label=label,
                    label_name=label_name,
                    attack_id=attack_id,
                    attack_group=attack_group,
                    track="19LA",
                    split=split,
                    metadata={"speaker": speaker},
                )
            )
    return samples


def _build_2021_la_samples(database_path: Path, subset: str) -> List[SampleRecord]:
    protocol_path = REPO_ROOT / "eval-package" / "21" / "keys" / "LA" / "CM" / "trial_metadata.txt"
    audio_dir = database_path / "ASVspoof2021_LA_eval" / "flac"
    samples: List[SampleRecord] = []
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            speaker, utt_id, codec, tx, attack_id, label_name, trim, subset_name = line.strip().split()
            if subset != "all" and subset_name.lower() != subset:
                continue
            label = 1 if label_name == "bonafide" else 0
            samples.append(
                SampleRecord(
                    utt_id=utt_id,
                    audio_path=audio_dir / f"{utt_id}.flac",
                    label=label,
                    label_name=label_name,
                    attack_id=attack_id,
                    attack_group=-1,
                    track="21LA",
                    split=subset_name.lower(),
                    metadata={
                        "speaker": speaker,
                        "codec": codec,
                        "transmission": tx,
                        "trim": trim,
                    },
                )
            )
    return samples


def _build_2021_df_samples(database_path: Path, subset: str) -> List[SampleRecord]:
    protocol_path = REPO_ROOT / "eval-package" / "21" / "keys" / "DF" / "CM" / "trial_metadata.txt"
    audio_dir = database_path / "ASVspoof2021_DF_eval" / "flac"
    samples: List[SampleRecord] = []
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            speaker, utt_id, codec, source_set, attack_id, label_name, trim, subset_name = parts[:8]
            if subset != "all" and subset_name.lower() != subset:
                continue
            extra = parts[8:]
            metadata = {
                "speaker": speaker,
                "codec": codec,
                "source_set": source_set,
                "trim": trim,
            }
            if len(extra) >= 1:
                metadata["vocoder_family"] = extra[0]
            if len(extra) >= 2:
                metadata["team_task"] = extra[1]
            if len(extra) >= 3:
                metadata["team_name"] = extra[2]
            if len(extra) >= 4:
                metadata["gender_pair"] = extra[3]
            if len(extra) >= 5:
                metadata["condition_tag"] = extra[4]
            label = 1 if label_name == "bonafide" else 0
            samples.append(
                SampleRecord(
                    utt_id=utt_id,
                    audio_path=audio_dir / f"{utt_id}.flac",
                    label=label,
                    label_name=label_name,
                    attack_id=attack_id,
                    attack_group=-1,
                    track="21DF",
                    split=subset_name.lower(),
                    metadata=metadata,
                )
            )
    return samples


def _build_itw_samples(database_path: Path) -> List[SampleRecord]:
    protocol_path = REPO_ROOT / "eval-package" / "ITW" / "keys" / "trial_metadata.txt"
    audio_dir = database_path / "release_in_the_wild"
    samples: List[SampleRecord] = []
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            _, utt_id, _, _, _, label_name = parts[:6]
            label = 1 if label_name == "bona-fide" else 0
            normalized_label = "bonafide" if label == 1 else "spoof"
            samples.append(
                SampleRecord(
                    utt_id=utt_id,
                    audio_path=audio_dir / utt_id,
                    label=label,
                    label_name=normalized_label,
                    attack_id="-",
                    attack_group=-1,
                    track="ITW",
                    split="eval",
                    metadata={},
                )
            )
    return samples


def infer_embeddings(trainer, loader: DataLoader, device: str) -> Dict[str, object]:
    trainer.eval()
    all_features: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device, non_blocking=True)
            feature_tensor = trainer.get_features(audio)
            features = feature_tensor.detach().cpu().numpy()
            logits = trainer.classifier(feature_tensor).detach().cpu()
            probs = torch.softmax(logits, dim=1).numpy()
            preds = np.argmax(probs, axis=1)

            all_features.append(features)
            all_logits.append(logits.numpy())

            for i, utt_id in enumerate(batch["utt_ids"]):
                metadata = dict(batch["metadata"][i])
                rows.append(
                    {
                        "utt_id": utt_id,
                        "label": int(batch["labels"][i].item()),
                        "label_name": batch["label_names"][i],
                        "pred_label": int(preds[i]),
                        "pred_label_name": "bonafide" if preds[i] == 1 else "spoof",
                        "score_spoof": float(probs[i, 0]),
                        "score_bonafide": float(probs[i, 1]),
                        "attack_id": batch["attack_ids"][i],
                        "attack_group": int(batch["attack_groups"][i].item()),
                        "track": batch["tracks"][i],
                        "split": batch["splits"][i],
                        **metadata,
                    }
                )

    return {
        "features": np.concatenate(all_features, axis=0),
        "logits": np.concatenate(all_logits, axis=0),
        "rows": rows,
    }


def save_feature_bundle(output_prefix: Path, bundle: Dict[str, object]) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": bundle["features"],
            "logits": bundle["logits"],
            "rows": bundle["rows"],
        },
        output_prefix.with_suffix(".pt"),
    )
    write_rows_csv(output_prefix.with_suffix(".csv"), bundle["rows"])


def write_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_attack_name(row: Dict[str, object], field_name: str) -> Optional[str]:
    value = row.get(field_name)
    if value is None:
        return None
    value = str(value)
    if value.strip() in {"", "-", "nan", "None"}:
        return None
    return value


def confidence_margin(score_spoof: float, score_bonafide: float) -> float:
    return abs(score_spoof - score_bonafide)


def topk_rows(rows: Iterable[Dict[str, object]], k: int, key: str, reverse: bool = True) -> List[Dict[str, object]]:
    return sorted(rows, key=lambda x: x[key], reverse=reverse)[:k]
