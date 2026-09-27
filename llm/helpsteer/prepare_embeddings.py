#!/usr/bin/env python3
"""Encode HelpSteer2 prompt/response pairs with a frozen sentence encoder."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/helpsteer2.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/helpsteer2_embeddings.npz"))
    parser.add_argument("--manifest", type=Path, default=Path("data/helpsteer2_embeddings_manifest.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_model_cache"))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and args.manifest.exists() and not args.force:
        print(args.manifest.read_text(encoding="utf-8"))
        return

    frame = pd.read_parquet(args.data)
    selected_roles = ["base_train", "pilot", "conformal", "test"]
    selected = frame.loc[frame["role"].isin(selected_roles)].copy()
    selected = selected.sort_values("row_id").reset_index(drop=True)
    if selected.empty or not selected["row_id"].is_unique:
        raise ValueError("Selected row IDs must be non-empty and unique")

    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = SentenceTransformer(args.model, cache_folder=str(cache_dir), device=device)
    model.max_seq_length = args.max_seq_length
    encode_kwargs = {
        "batch_size": args.batch_size,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
    prompt_embeddings = model.encode(selected["prompt"].tolist(), **encode_kwargs)
    response_embeddings = model.encode(selected["response"].tolist(), **encode_kwargs)
    features = np.concatenate([prompt_embeddings, response_embeddings], axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Embeddings contain non-finite values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        row_ids=selected["row_id"].to_numpy(dtype=str),
        roles=selected["role"].to_numpy(dtype=str),
        features=features,
    )
    manifest = {
        "model": args.model,
        "device": device,
        "max_seq_length": args.max_seq_length,
        "rows": int(len(selected)),
        "feature_dimension": int(features.shape[1]),
        "role_counts": {str(k): int(v) for k, v in selected["role"].value_counts().items()},
        "output_path": str(args.output.resolve()),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
