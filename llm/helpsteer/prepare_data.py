#!/usr/bin/env python3
"""Prepare leakage-safe HelpSteer2 roles at the prompt-group level."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset


LABELS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/helpsteer2.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("data/helpsteer2_manifest.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_datasets_cache"))
    parser.add_argument("--base-target", type=int, default=800)
    parser.add_argument("--pilot-target", type=int, default=500)
    parser.add_argument("--conformal-target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def prompt_key(prompt: str) -> str:
    normalized = normalize_text(prompt)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def allocate_prompt_groups(train: pd.DataFrame, target: int, rng: np.random.Generator) -> tuple[set[str], pd.DataFrame]:
    group_sizes = train.groupby("prompt_key", sort=False).size()
    keys = group_sizes.index.to_numpy().copy()
    rng.shuffle(keys)
    selected: list[str] = []
    rows = 0
    for key in keys:
        if rows >= target:
            break
        selected.append(str(key))
        rows += int(group_sizes.loc[key])
    selected_set = set(selected)
    return selected_set, train.loc[~train["prompt_key"].isin(selected_set)].copy()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    if output.exists() and manifest_path.exists() and not args.force:
        print(f"Using existing data: {output}")
        print(manifest_path.read_text(encoding="utf-8"))
        return

    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("nvidia/HelpSteer2", cache_dir=str(cache_dir))
    train = dataset["train"].to_pandas()
    validation = dataset["validation"].to_pandas()
    train["official_split"] = "train"
    validation["official_split"] = "validation"
    source_rows = len(train) + len(validation)

    frame = pd.concat([train, validation], ignore_index=True)
    frame["prompt_key"] = frame["prompt"].map(prompt_key)
    frame["normalized_response"] = frame["response"].map(normalize_text)
    duplicate_mask = frame.duplicated(["official_split", "prompt_key", "normalized_response"], keep="first")
    duplicate_rows_removed = int(duplicate_mask.sum())
    frame = frame.loc[~duplicate_mask].reset_index(drop=True)

    train = frame.loc[frame["official_split"].eq("train")].copy()
    validation = frame.loc[frame["official_split"].eq("validation")].copy()
    prompt_overlap = set(train["prompt_key"]).intersection(set(validation["prompt_key"]))
    if prompt_overlap:
        train = train.loc[~train["prompt_key"].isin(prompt_overlap)].copy()

    rng = np.random.default_rng(args.seed)
    base_keys, remaining = allocate_prompt_groups(train, args.base_target, rng)
    pilot_keys, remaining = allocate_prompt_groups(remaining, args.pilot_target, rng)
    conformal_keys, remaining = allocate_prompt_groups(remaining, args.conformal_target, rng)

    train["role"] = "pool"
    train.loc[train["prompt_key"].isin(base_keys), "role"] = "base_train"
    train.loc[train["prompt_key"].isin(pilot_keys), "role"] = "pilot"
    train.loc[train["prompt_key"].isin(conformal_keys), "role"] = "conformal"
    validation["role"] = "test"
    result = pd.concat([train, validation], ignore_index=True)
    result.insert(0, "row_id", [f"hs2-{index:05d}" for index in range(len(result))])
    result["prompt_chars"] = result["prompt"].str.len()
    result["response_chars"] = result["response"].str.len()

    required = ["row_id", "prompt", "response", *LABELS, "official_split", "prompt_key", "role"]
    if result[required].isna().any().any():
        raise ValueError("Required fields contain nulls")
    if not result["row_id"].is_unique:
        raise ValueError("row_id is not unique")
    if any(not result[label].between(0, 4).all() for label in LABELS):
        raise ValueError("Human ratings must lie in [0, 4]")
    role_prompt_counts = result.groupby("prompt_key")["role"].nunique()
    if int(role_prompt_counts.gt(1).sum()) != 0:
        raise ValueError("A normalized prompt appears in multiple experiment roles")

    output.parent.mkdir(parents=True, exist_ok=True)
    result.drop(columns=["normalized_response"]).to_parquet(output, index=False)
    manifest = {
        "dataset": "nvidia/HelpSteer2",
        "seed": args.seed,
        "source_rows": int(source_rows),
        "rows_after_deduplication": int(len(result)),
        "duplicate_prompt_response_rows_removed": duplicate_rows_removed,
        "train_validation_prompt_overlap_removed_from_train": int(len(prompt_overlap)),
        "role_counts": {str(k): int(v) for k, v in result["role"].value_counts().items()},
        "role_prompt_counts": {
            str(k): int(v) for k, v in result.groupby("role")["prompt_key"].nunique().items()
        },
        "cross_role_prompt_overlap": int(role_prompt_counts.gt(1).sum()),
        "missing_required_values": int(result[required].isna().sum().sum()),
        "out_of_range_label_values": int(
            sum((~result[label].between(0, 4)).sum() for label in LABELS)
        ),
        "label_summary": result[LABELS].describe().round(6).to_dict(),
        "text_length_summary": result[["prompt_chars", "response_chars"]].describe().round(3).to_dict(),
        "output_path": str(output),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
