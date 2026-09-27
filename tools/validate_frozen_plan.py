#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from synrcp_v11.plan import load_plan, validate_plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the frozen V11 experiment plan")
    parser.add_argument("plan", nargs="?", default="plans/all_tasks.jsonl")
    args = parser.parse_args()
    rows = load_plan(Path(args.plan))
    errors = validate_plan(rows)
    if errors:
        print("V11 PLAN INVALID")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"V11 PLAN VALID: {len(rows)} unique tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
