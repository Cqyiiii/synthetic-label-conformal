#!/usr/bin/env python3
"""Prepare a deterministic Civil Comments sample for the DeepSeek pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset


DEFAULT_SEED = 20260803


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/civil_comments_20k.parquet"))
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--base-train-size", type=int, default=400)
    parser.add_argument("--pilot-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_frame(frame: pd.DataFrame, expected_rows: int) -> None:
    required = {"row_id", "text", "toxicity", "role"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(frame)}")
    if frame["row_id"].duplicated().any():
        raise ValueError("row_id is not unique")
    if frame["text"].isna().any() or frame["text"].str.strip().eq("").any():
        raise ValueError("Found missing or empty comment text")
    if not frame["toxicity"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("toxicity values must lie in [0, 1]")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.force:
        frame = pd.read_parquet(output)
        validate_frame(frame, args.sample_size)
        print(f"Using existing sample: {output}")
        print(frame["role"].value_counts().to_string())
        return

    if args.base_train_size + args.pilot_size >= args.sample_size:
        raise ValueError("base-train-size + pilot-size must be smaller than sample-size")

    dataset = load_dataset("google/civil_comments", split="validation")
    if len(dataset) < args.sample_size:
        raise ValueError(f"Requested {args.sample_size} rows from a split with {len(dataset)} rows")

    sampled = dataset.shuffle(seed=args.seed).select(range(args.sample_size))
    frame = sampled.to_pandas()
    keep = [column for column in ["id", "text", "toxicity"] if column in frame.columns]
    frame = frame.loc[:, keep].copy()
    if "id" in frame.columns:
        frame = frame.rename(columns={"id": "source_id"})

    frame.insert(0, "row_id", [f"cc-{idx:05d}" for idx in range(len(frame))])
    role = np.full(len(frame), "pool", dtype=object)
    role[: args.base_train_size] = "base_train"
    pilot_start = args.base_train_size
    pilot_end = pilot_start + args.pilot_size
    role[pilot_start:pilot_end] = "pilot"
    frame["role"] = role

    validate_frame(frame, args.sample_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)

    duplicate_text_rate = float(frame["text"].duplicated().mean())
    print(f"Saved deterministic sample: {output}")
    print(frame["role"].value_counts().to_string())
    print(
        "toxicity summary: "
        f"mean={frame['toxicity'].mean():.4f}, "
        f"std={frame['toxicity'].std():.4f}, "
        f">=0.5={frame['toxicity'].ge(0.5).mean():.3%}, "
        f"duplicate_text_rate={duplicate_text_rate:.3%}"
    )


if __name__ == "__main__":
    main()
