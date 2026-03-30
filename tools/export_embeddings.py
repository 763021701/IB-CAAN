import argparse
from pathlib import Path

from analysis_common import (
    build_analysis_loader,
    build_trainer_from_config,
    infer_embeddings,
    load_model_weights,
    save_feature_bundle,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export frozen embeddings and logits for probe analysis.")
    parser.add_argument("--config", required=True, type=str, help="Path to experiment config JSON.")
    parser.add_argument("--database-path", required=True, type=str, help="Root path of datasets.")
    parser.add_argument("--track", required=True, choices=["19LA", "21LA", "21DF", "ITW"], help="Dataset track.")
    parser.add_argument("--split", default="eval", choices=["train", "dev", "eval"], help="19LA split.")
    parser.add_argument("--subset", default="all", choices=["all", "eval", "progress"], help="Subset filter for ASVspoof2021.")
    parser.add_argument("--batch-size", default=None, type=int, help="Override inference batch size.")
    parser.add_argument("--num-workers", default=None, type=int, help="Override dataloader workers.")
    parser.add_argument("--output-prefix", required=True, type=str, help="Output prefix without suffix.")
    return parser.parse_args()


def main():
    args = parse_args()
    config, trainer, device = build_trainer_from_config(args.config)
    ckpt_path = load_model_weights(trainer, config)
    _, samples, loader = build_analysis_loader(
        args.config,
        args.database_path,
        track=args.track,
        split=args.split,
        subset=args.subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    bundle = infer_embeddings(trainer, loader, device)
    for row in bundle["rows"]:
        row["checkpoint_path"] = ckpt_path
        row["feature_source"] = Path(args.output_prefix).name

    save_feature_bundle(Path(args.output_prefix), bundle)
    print(f"Exported {len(samples)} samples to {args.output_prefix}.pt/.csv")


if __name__ == "__main__":
    main()
