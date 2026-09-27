#!/usr/bin/env python3
"""Generate resumable DeepSeek toxicity labels for the Civil Comments pilot."""

from __future__ import annotations

import argparse
import asyncio
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

SYSTEM_PROMPT = """You are an independent content-moderation annotator.
Estimate the fraction of independent human moderators who would judge a comment as toxic.
Toxic means rude, disrespectful, or unreasonable language likely to make someone leave a discussion.
Return JSON only in exactly this form: {"toxicity": 0.0}
The toxicity value must be a number between 0 and 1. Do not explain the answer and do not repeat the comment."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/civil_comments_20k.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_ALIASES), default=["flash", "pro"])
    parser.add_argument("--role", default="pilot")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=32)
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


async def label_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    model_alias: str,
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
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Comment:\n{row['text']}"},
                    ],
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
            toxicity = parse_toxicity(content)
            reasoning_content = getattr(message, "reasoning_content", None)
            return {
                "row_id": row["row_id"],
                "model_alias": model_alias,
                "requested_model": MODEL_ALIASES[model_alias],
                "served_model": response.model,
                "response_id": response.id,
                "status": "ok",
                "toxicity_pred": toxicity,
                "raw_content": content,
                "reasoning_content_present": bool(reasoning_content),
                "attempts": attempt,
                "latency_seconds": latency,
                "cumulative_latency_seconds": cumulative_latency,
                "errors": errors,
                **aggregate_usage,
            }
        except Exception as exc:  # API and validation failures are both retryable here.
            cumulative_latency += time.perf_counter() - started
            errors.append(f"{type(exc).__name__}: {str(exc)[:300]}")
            if attempt < max_attempts:
                delay = min(20.0, (2 ** (attempt - 1)) + random.random())
                await asyncio.sleep(delay)

    return {
        "row_id": row["row_id"],
        "model_alias": model_alias,
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


async def run_model(
    client: AsyncOpenAI,
    rows: list[dict[str, Any]],
    output_path: Path,
    model_alias: str,
    concurrency: int,
    max_attempts: int,
    max_tokens: int,
) -> None:
    completed = load_completed(output_path)
    pending = [row for row in rows if str(row["row_id"]) not in completed]
    print(f"{model_alias}: {len(completed)} completed, {len(pending)} pending")
    if not pending:
        return

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(
            label_one(client, semaphore, row, model_alias, max_attempts, max_tokens)
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
            if record["status"] == "ok":
                ok += 1
            else:
                errors += 1
            if index % 25 == 0 or index == len(tasks):
                elapsed = time.perf_counter() - started
                print(
                    f"{model_alias}: {index}/{len(tasks)} new rows, "
                    f"ok={ok}, error={errors}, elapsed={elapsed:.1f}s",
                    flush=True,
                )


async def async_main(args: argparse.Namespace) -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    frame = pd.read_parquet(args.data.resolve())
    frame = frame.loc[frame["role"].eq(args.role), ["row_id", "text"]].copy()
    if args.limit is not None:
        frame = frame.head(args.limit)
    rows = frame.to_dict(orient="records")
    if not rows:
        raise ValueError(f"No rows found for role={args.role!r}")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=args.request_timeout,
        max_retries=0,
    )
    for model_alias in args.models:
        await run_model(
            client=client,
            rows=rows,
            output_path=args.output_dir.resolve() / f"{model_alias}_labels.jsonl",
            model_alias=model_alias,
            concurrency=args.concurrency,
            max_attempts=args.max_attempts,
            max_tokens=args.max_tokens,
        )


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
