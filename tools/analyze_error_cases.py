import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from analysis_common import (
    build_analysis_loader,
    build_trainer_from_config,
    confidence_margin,
    infer_embeddings,
    load_model_weights,
    topk_rows,
    write_rows_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export sample-level predictions and error-case summaries.")
    parser.add_argument("--config", required=True, type=str, help="Path to experiment config JSON.")
    parser.add_argument("--database-path", required=True, type=str, help="Root path of datasets.")
    parser.add_argument("--track", required=True, choices=["19LA", "21LA", "21DF", "ITW"], help="Dataset track.")
    parser.add_argument("--split", default="eval", choices=["train", "dev", "eval"], help="19LA split.")
    parser.add_argument("--subset", default="all", choices=["all", "eval", "progress"], help="Subset filter for ASVspoof2021.")
    parser.add_argument("--batch-size", default=None, type=int, help="Override inference batch size.")
    parser.add_argument("--num-workers", default=None, type=int, help="Override dataloader workers.")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory to store analysis outputs.")
    parser.add_argument("--top-k", default=10, type=int, help="Number of representative cases per category.")
    return parser.parse_args()


def enrich_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    enriched = []
    for row in rows:
        item = dict(row)
        item["is_error"] = int(item["label"] != item["pred_label"])
        item["error_type"] = "correct"
        if item["label"] == 1 and item["pred_label"] == 0:
            item["error_type"] = "false_positive"
        elif item["label"] == 0 and item["pred_label"] == 1:
            item["error_type"] = "false_negative"
        item["confidence"] = max(float(item["score_spoof"]), float(item["score_bonafide"]))
        item["margin"] = confidence_margin(float(item["score_spoof"]), float(item["score_bonafide"]))
        enriched.append(item)
    return enriched


def aggregate_by_field(rows: List[Dict[str, object]], field: str) -> List[Dict[str, object]]:
    bucket = defaultdict(lambda: {"count": 0, "errors": 0, "false_positive": 0, "false_negative": 0})
    for row in rows:
        key = row.get(field, "NA")
        key = "NA" if key in [None, "", "-", "nan"] else str(key)
        bucket[key]["count"] += 1
        bucket[key]["errors"] += int(row["is_error"])
        bucket[key]["false_positive"] += int(row["error_type"] == "false_positive")
        bucket[key]["false_negative"] += int(row["error_type"] == "false_negative")

    summary = []
    for key, stats in bucket.items():
        count = max(stats["count"], 1)
        summary.append(
            {
                field: key,
                "count": stats["count"],
                "errors": stats["errors"],
                "error_rate": stats["errors"] / count,
                "false_positive": stats["false_positive"],
                "false_negative": stats["false_negative"],
            }
        )
    return sorted(summary, key=lambda x: (-x["errors"], x[field]))


def build_case_tables(rows: List[Dict[str, object]], top_k: int) -> Dict[str, List[Dict[str, object]]]:
    false_positive = [row for row in rows if row["error_type"] == "false_positive"]
    false_negative = [row for row in rows if row["error_type"] == "false_negative"]
    boundary_errors = [row for row in rows if row["is_error"] == 1]

    return {
        "false_positive_high_conf": topk_rows(false_positive, top_k, key="confidence", reverse=True),
        "false_negative_high_conf": topk_rows(false_negative, top_k, key="confidence", reverse=True),
        "boundary_errors_low_margin": topk_rows(boundary_errors, top_k, key="margin", reverse=False),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config, trainer, device = build_trainer_from_config(args.config)
    ckpt_path = load_model_weights(trainer, config)
    _, _, loader = build_analysis_loader(
        args.config,
        args.database_path,
        track=args.track,
        split=args.split,
        subset=args.subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    bundle = infer_embeddings(trainer, loader, device)
    rows = enrich_rows(bundle["rows"])
    for row in rows:
        row["checkpoint_path"] = ckpt_path

    write_rows_csv(output_dir / "sample_scores.csv", rows)
    write_rows_csv(output_dir / "error_by_attack.csv", aggregate_by_field(rows, "attack_id"))
    write_rows_csv(output_dir / "error_by_attack_group.csv", aggregate_by_field(rows, "attack_group"))

    optional_fields = ["codec", "transmission", "source_set", "vocoder_family", "team_task", "team_name", "condition_tag"]
    for field in optional_fields:
        if any(field in row for row in rows):
            write_rows_csv(output_dir / f"error_by_{field}.csv", aggregate_by_field(rows, field))

    case_tables = build_case_tables(rows, args.top_k)
    for name, case_rows in case_tables.items():
        write_rows_csv(output_dir / f"{name}.csv", case_rows)

    summary = {
        "num_samples": len(rows),
        "num_errors": sum(int(row["is_error"]) for row in rows),
        "false_positive": sum(int(row["error_type"] == "false_positive") for row in rows),
        "false_negative": sum(int(row["error_type"] == "false_negative") for row in rows),
        "track": args.track,
        "split": args.split,
        "subset": args.subset,
        "checkpoint_path": ckpt_path,
    }
    with open(output_dir / "error_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved error analysis to {output_dir}")


if __name__ == "__main__":
    main()
