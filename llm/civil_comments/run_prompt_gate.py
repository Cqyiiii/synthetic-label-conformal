#!/usr/bin/env python3
"""Run the zero-shot versus ICL prompt-calibration gate on DeepSeek."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI


MODEL_ALIASES = {
    "flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
}
PROMPT_VARIANTS = ("zero_shot", "icl", "icl_distributional")
BASE_SYSTEM_PROMPT = """You are an independent content-moderation annotator.
Estimate the fraction of independent human moderators who would judge a comment as toxic.
Toxic means rude, disrespectful, or unreasonable language likely to make someone leave a discussion.
Return JSON only in exactly this form: {"toxicity": 0.0}
The toxicity value must be a number between 0 and 1. Do not explain the answer and do not repeat the comment."""
DISTRIBUTIONAL_INSTRUCTION = """

The target is the expected fraction of independent annotators who would mark the comment as toxic, not an abstract severity rating. Most naturally sampled comments receive zero or near-zero scores. Match the empirical human-vote scale illustrated by the examples."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=Path("prompt_gate/data/prompt_gate_assignments.parquet")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("prompt_gate/results"))
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_ALIASES), default=["flash", "pro"])
    parser.add_argument("--variants", nargs="+", choices=PROMPT_VARIANTS, default=list(PROMPT_VARIANTS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt-seed", type=int, default=20260804)
    parser.add_argument("--run-tag", default="interleaved_v2")
    return parser.parse_args()


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok":
                completed.add(str(record["row_id"]))
    return completed


def usage_to_dict(response: Any) -> dict[str, int]:
    usage = response.usage
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
        }
    raw = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    return {
        "prompt_tokens": int(raw.get("prompt_tokens") or 0),
        "completion_tokens": int(raw.get("completion_tokens") or 0),
        "prompt_cache_hit_tokens": int(raw.get("prompt_cache_hit_tokens") or 0),
        "prompt_cache_miss_tokens": int(raw.get("prompt_cache_miss_tokens") or 0),
    }


def parse_toxicity(content: str) -> float:
    if not content or not content.strip():
        raise ValueError("empty response content")
    payload = json.loads(content)
    if "toxicity" not in payload:
        raise ValueError("JSON response has no toxicity field")
    value = float(payload["toxicity"])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"toxicity is outside [0, 1]: {value}")
    return value


def build_messages(variant: str, comment: str, examples: list[dict[str, Any]]) -> list[dict[str, str]]:
    system_prompt = BASE_SYSTEM_PROMPT
    if variant == "icl_distributional":
        system_prompt += DISTRIBUTIONAL_INSTRUCTION
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if variant != "zero_shot":
        for example in examples:
            messages.extend(
                [
                    {"role": "user", "content": f"Comment:\n{example['text']}"},
                    {
                        "role": "assistant",
                        "content": json.dumps({"toxicity": round(float(example["toxicity"]), 6)}),
                    },
                ]
            )
    messages.append({"role": "user", "content": f"Comment:\n{comment}"})
    return messages


async def label_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    examples: list[dict[str, Any]],
    model_alias: str,
    variant: str,
    prompt_order_hash: str,
    max_attempts: int,
    max_tokens: int,
) -> dict[str, Any]:
    aggregate_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
    }
    errors: list[str] = []
    cumulative_latency = 0.0

    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=MODEL_ALIASES[model_alias],
                    messages=build_messages(variant, str(row["text"]), examples),
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=max_tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                )
            latency = time.perf_counter() - started
            cumulative_latency += latency
            usage = usage_to_dict(response)
            for key, value in usage.items():
                aggregate_usage[key] += value
            message = response.choices[0].message
            content = message.content or ""
            return {
                "row_id": row["row_id"],
                "model_alias": model_alias,
                "prompt_variant": variant,
                "prompt_order_hash": prompt_order_hash,
                "requested_model": MODEL_ALIASES[model_alias],
                "served_model": response.model,
                "response_id": response.id,
                "status": "ok",
                "toxicity_pred": parse_toxicity(content),
                "raw_content": content,
                "reasoning_content_present": bool(getattr(message, "reasoning_content", None)),
                "attempts": attempt,
                "latency_seconds": latency,
                "cumulative_latency_seconds": cumulative_latency,
                "errors": errors,
                **aggregate_usage,
            }
        except Exception as exc:
            cumulative_latency += time.perf_counter() - started
            errors.append(f"{type(exc).__name__}: {str(exc)[:300]}")
            if attempt < max_attempts:
                await asyncio.sleep(min(20.0, (2 ** (attempt - 1)) + random.random()))

    return {
        "row_id": row["row_id"],
        "model_alias": model_alias,
        "prompt_variant": variant,
        "prompt_order_hash": prompt_order_hash,
        "requested_model": MODEL_ALIASES[model_alias],
        "served_model": None,
        "response_id": None,
        "status": "error",
        "toxicity_pred": None,
        "raw_content": None,
        "reasoning_content_present": False,
        "attempts": max_attempts,
        "latency_seconds": None,
        "cumulative_latency_seconds": cumulative_latency,
        "errors": errors,
        **aggregate_usage,
    }


async def run_combination(
    client: AsyncOpenAI,
    rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    output_path: Path,
    model_alias: str,
    variant: str,
    prompt_order_hash: str,
    concurrency: int,
    max_attempts: int,
    max_tokens: int,
) -> None:
    completed = load_completed(output_path)
    pending = [row for row in rows if str(row["row_id"]) not in completed]
    print(f"{model_alias}/{variant}: {len(completed)} completed, {len(pending)} pending")
    if not pending:
        return

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(
            label_one(
                client,
                semaphore,
                row,
                examples,
                model_alias,
                variant,
                prompt_order_hash,
                max_attempts,
                max_tokens,
            )
        )
        for row in pending
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ok = 0
    errors = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for index, task in enumerate(asyncio.as_completed(tasks), start=1):
            record = await task
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            ok += int(record["status"] == "ok")
            errors += int(record["status"] != "ok")
            if index % 25 == 0 or index == len(tasks):
                elapsed = time.perf_counter() - started
                print(
                    f"{model_alias}/{variant}: {index}/{len(tasks)} new rows, "
                    f"ok={ok}, error={errors}, elapsed={elapsed:.1f}s",
                    flush=True,
                )


async def async_main(args: argparse.Namespace) -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    frame = pd.read_parquet(args.data.resolve())
    examples = frame.loc[
        frame["prompt_role"].eq("prompt_dev"), ["row_id", "text", "toxicity", "true_bin"]
    ].copy()
    # Interleave label bins deterministically. Grouping zero-score examples at the
    # end creates a recency confound and is not a valid ICL comparison.
    examples = examples.sample(frac=1.0, random_state=args.prompt_seed).reset_index(drop=True)
    prompt_order_hash = hashlib.sha256(
        "|".join(examples["row_id"].astype(str)).encode("utf-8")
    ).hexdigest()[:16]
    examples = examples.to_dict(orient="records")
    validation = frame.loc[frame["prompt_role"].eq("prompt_validation"), ["row_id", "text"]]
    validation = validation.sort_values("row_id")
    if args.limit is not None:
        validation = validation.head(args.limit)
    rows = validation.to_dict(orient="records")
    if len(examples) != 16:
        raise ValueError(f"Expected 16 prompt-development examples, found {len(examples)}")
    if not rows:
        raise ValueError("No prompt-validation rows found")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=args.request_timeout,
        max_retries=0,
    )
    for model_alias in args.models:
        for variant in args.variants:
            await run_combination(
                client=client,
                rows=rows,
                examples=examples,
                output_path=args.output_dir.resolve()
                / f"{model_alias}_{variant}_{args.run_tag}.jsonl",
                model_alias=model_alias,
                variant=variant,
                prompt_order_hash=prompt_order_hash,
                concurrency=args.concurrency,
                max_attempts=args.max_attempts,
                max_tokens=args.max_tokens,
            )


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
