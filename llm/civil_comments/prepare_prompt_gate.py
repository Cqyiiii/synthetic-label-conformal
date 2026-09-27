#!/usr/bin/env python3
"""Create leakage-free prompt-development and prompt-validation splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEED = 20260804
DEV_COUNTS = {
    "zero_near": 8,
    "low": 3,
    "mid": 2,
    "high": 2,
    "very_high": 1,
}
IDENTITY_PATTERN = re.compile(
    r"\b(?:woman|women|man|men|girl|boy|black|white|asian|latino|latina|"
    r"gay|lesbian|trans|transgender|muslim|christian|jew|jewish|hindu|"
    r"immigrant|disabled|disability)\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/civil_comments_20k.parquet"))
    parser.add_argument(
        "--output", type=Path, default=Path("prompt_gate/data/prompt_gate_assignments.parquet")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("prompt_gate/data/prompt_gate_manifest.json")
    )
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def true_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-1e-12, 0.001, 0.2, 0.5, 0.8, 1.000001],
        labels=["zero_near", "low", "mid", "high", "very_high"],
        include_lowest=True,
    ).astype("object")


def stable_jitter(row_id: str, seed: int) -> float:
    payload = f"{seed}:{row_id}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:12], 16) / float(16**12)


def select_diverse_examples(pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    selected: list[pd.Series] = []
    identity_targets = {
        "zero_near": [False, False, False, False, False, False, False, True],
        "low": [False, False, True],
        "mid": [False, True],
        "high": [False, True],
        "very_high": [True],
    }

    for bin_index, (bin_name, count) in enumerate(DEV_COUNTS.items()):
        candidates = pool.loc[pool["true_bin"].eq(bin_name)].copy()
        candidates = candidates.loc[candidates["char_len"].between(20, 1_200)].copy()
        if len(candidates) < count:
            raise ValueError(f"Insufficient candidates in {bin_name}: {len(candidates)}")
        candidates["length_rank"] = candidates["char_len"].rank(pct=True, method="average")
        candidates["jitter"] = [stable_jitter(row_id, seed + bin_index) for row_id in candidates["row_id"]]

        chosen_ids: set[str] = set()
        target_ranks = np.linspace(0.12, 0.88, count)
        for target_rank, want_identity in zip(target_ranks, identity_targets[bin_name]):
            remaining = candidates.loc[~candidates["row_id"].isin(chosen_ids)].copy()
            identity_penalty = (remaining["identity_mention"] != want_identity).astype(float) * 0.35
            remaining["selection_score"] = (
                (remaining["length_rank"] - target_rank).abs()
                + identity_penalty
                + 0.01 * remaining["jitter"]
            )
            choice = remaining.sort_values(["selection_score", "row_id"]).iloc[0]
            chosen_ids.add(str(choice["row_id"]))
            selected.append(choice)

    result = pd.DataFrame(selected).drop(columns=["length_rank", "jitter", "selection_score"], errors="ignore")
    if len(result) != sum(DEV_COUNTS.values()) or result["row_id"].duplicated().any():
        raise AssertionError("Prompt-development selection is not unique or complete")
    return result


def assign_calibration_folds(validation: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    validation = validation.copy()
    validation["calibration_fold"] = "evaluation"
    train_indices: list[int] = []
    for _, group in validation.groupby("true_bin", sort=False):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        train_indices.extend(indices[: len(indices) // 2].tolist())

    target = len(validation) // 2
    train_set = set(train_indices)
    if len(train_set) < target:
        remaining = np.array(sorted(set(validation.index) - train_set))
        rng.shuffle(remaining)
        train_set.update(remaining[: target - len(train_set)].tolist())
    elif len(train_set) > target:
        removable = np.array(sorted(train_set))
        rng.shuffle(removable)
        for index in removable[: len(train_set) - target]:
            train_set.remove(int(index))
    validation.loc[list(train_set), "calibration_fold"] = "calibration_train"
    return validation


def validate_assignments(assignments: pd.DataFrame, source: pd.DataFrame, validation_size: int) -> dict:
    dev = assignments.loc[assignments["prompt_role"].eq("prompt_dev")]
    validation = assignments.loc[assignments["prompt_role"].eq("prompt_validation")]
    historical = source.loc[source["role"].isin(["base_train", "pilot"])].copy()
    historical_norm = set(historical["normalized_text"])

    checks = {
        "rows": int(len(assignments)),
        "prompt_dev_rows": int(len(dev)),
        "prompt_validation_rows": int(len(validation)),
        "unique_row_ids": bool(assignments["row_id"].is_unique),
        "unique_normalized_text": bool(assignments["normalized_text"].is_unique),
        "overlap_with_historical_row_ids": int(
            assignments["row_id"].isin(set(historical["row_id"])).sum()
        ),
        "overlap_with_historical_text": int(assignments["normalized_text"].isin(historical_norm).sum()),
        "dev_validation_row_overlap": int(
            len(set(dev["row_id"]).intersection(set(validation["row_id"])))
        ),
        "missing_required_values": int(
            assignments[["row_id", "text", "toxicity", "prompt_role"]].isna().sum().sum()
        ),
        "toxicity_out_of_range": int((~assignments["toxicity"].between(0, 1)).sum()),
        "validation_fold_counts": {
            str(key): int(value) for key, value in validation["calibration_fold"].value_counts().items()
        },
        "dev_bin_counts": {
            str(key): int(value) for key, value in dev["true_bin"].value_counts().items()
        },
        "validation_bin_counts": {
            str(key): int(value) for key, value in validation["true_bin"].value_counts().items()
        },
        "dev_identity_mentions": int(dev["identity_mention"].sum()),
        "validation_identity_mention_rate": float(validation["identity_mention"].mean()),
        "validation_toxicity_mean": float(validation["toxicity"].mean()),
        "validation_toxicity_ge_0_5_rate": float(validation["toxicity"].ge(0.5).mean()),
    }
    expected = {
        "rows": validation_size + sum(DEV_COUNTS.values()),
        "prompt_dev_rows": sum(DEV_COUNTS.values()),
        "prompt_validation_rows": validation_size,
        "unique_row_ids": True,
        "unique_normalized_text": True,
        "overlap_with_historical_row_ids": 0,
        "overlap_with_historical_text": 0,
        "dev_validation_row_overlap": 0,
        "missing_required_values": 0,
        "toxicity_out_of_range": 0,
    }
    failures = {key: (checks[key], value) for key, value in expected.items() if checks[key] != value}
    if failures:
        raise ValueError(f"Prompt-gate assignment validation failed: {failures}")
    return checks


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    if output.exists() and manifest_path.exists() and not args.force:
        print(f"Using existing assignments: {output}")
        print(manifest_path.read_text(encoding="utf-8"))
        return

    source = pd.read_parquet(args.data.resolve()).copy()
    required = {"row_id", "text", "toxicity", "role"}
    if missing := required.difference(source.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    source["normalized_text"] = source["text"].map(normalize_text)
    source["char_len"] = source["text"].str.len()
    source["identity_mention"] = source["text"].str.contains(IDENTITY_PATTERN, na=False)
    source["true_bin"] = true_bin(source["toxicity"])

    historical_norm = set(source.loc[source["role"].isin(["base_train", "pilot"]), "normalized_text"])
    pool = source.loc[source["role"].eq("pool")].copy()
    pool = pool.loc[~pool["normalized_text"].isin(historical_norm)].copy()
    pool = pool.sort_values("row_id").drop_duplicates("normalized_text", keep="first")

    dev = select_diverse_examples(pool, args.seed)
    remaining = pool.loc[~pool["row_id"].isin(set(dev["row_id"]))].copy()
    validation = remaining.sample(n=args.validation_size, random_state=args.seed).copy()
    validation = validation.sort_values("row_id").reset_index(drop=True)
    validation = assign_calibration_folds(validation, args.seed + 1)

    dev["prompt_role"] = "prompt_dev"
    dev["calibration_fold"] = "not_applicable"
    validation["prompt_role"] = "prompt_validation"
    assignments = pd.concat([dev, validation], ignore_index=True)
    assignments = assignments[
        [
            "row_id",
            "text",
            "toxicity",
            "prompt_role",
            "calibration_fold",
            "true_bin",
            "char_len",
            "identity_mention",
            "normalized_text",
        ]
    ].copy()
    checks = validate_assignments(assignments, source, args.validation_size)

    output.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(output, index=False)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "source_path": str(args.data.resolve()),
        "output_path": str(output),
        "prompt_dev_requested_bin_counts": DEV_COUNTS,
        "quality_checks": checks,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
