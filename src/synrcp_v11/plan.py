from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .registry import ABLATION_METHODS, FORBIDDEN_MAIN_NAMES, FORBIDDEN_MAIN_PREFIXES, MAIN_METHODS

EXPECTED_SUITE_COUNTS: dict[str, int] = {
    "v11_main": 240,
    "v11_teacher_quality": 160,
    "v11_pool_size": 160,
    "v11_trusted_budget": 80,
    "v11_ablation": 100,
}


def load_plan(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number} of {path} is not a JSON object")
        rows.append(row)
    return rows


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def validate_plan(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    task_ids = [str(row.get("task_id", "")) for row in rows]
    if any(not task_id for task_id in task_ids):
        errors.append("Every task must have a nonempty task_id")
    duplicate_ids = _duplicates(task_ids)
    if duplicate_ids:
        errors.append(f"Duplicate task IDs: {duplicate_ids[:5]}")

    counts = Counter(str(row.get("suite")) for row in rows)
    if dict(counts) != EXPECTED_SUITE_COUNTS:
        errors.append(f"Suite counts {dict(counts)} do not match {EXPECTED_SUITE_COUNTS}")

    for row in rows:
        suite = row.get("suite")
        methods = tuple(row.get("methods", []))
        task_id = row.get("task_id", "<unknown>")
        if suite == "v11_main" and methods != MAIN_METHODS:
            errors.append(f"{task_id}: main methods do not match the frozen registry")
        if suite == "v11_ablation" and methods != ABLATION_METHODS:
            errors.append(f"{task_id}: ablation methods do not match the frozen registry")
        if suite == "v11_main":
            for method in methods:
                if method in FORBIDDEN_MAIN_NAMES or any(method.startswith(p) for p in FORBIDDEN_MAIN_PREFIXES):
                    errors.append(f"{task_id}: forbidden internal method {method}")
        if not row.get("exact_conformal_order_statistic", False):
            errors.append(f"{task_id}: exact conformal order statistic is required")
        policy = row.get("dataset_sample_policy", {})
        if policy.get("default") != "full_cleaned" or policy.get("sgemm_max_rows") != 50000:
            errors.append(f"{task_id}: dataset sample policy is not frozen")
        metrics = row.get("metrics", {})
        if metrics.get("main_grouped_msce", {}).get("k") != 30:
            errors.append(f"{task_id}: main grouped MSCE must use K=30")
        if metrics.get("audit_grouped_msce", {}).get("k") != 10:
            errors.append(f"{task_id}: audit grouped MSCE must use K=10")
        if row.get("dataset") == "wec" and suite == "v11_main":
            if "CQR" not in methods or "CQR-PPI" not in methods:
                errors.append(f"{task_id}: WEC must include CQR and CQR-PPI")
    return errors
