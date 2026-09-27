from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Iterable

from .io import read_json, stable_hash


def _values(spec: Any) -> list[Any]:
    if isinstance(spec, list):
        return spec
    if isinstance(spec, dict) and "range" in spec:
        start, stop = spec["range"][:2]
        step = spec["range"][2] if len(spec["range"]) == 3 else 1
        return list(range(int(start), int(stop), int(step)))
    if isinstance(spec, dict) and "linspace" in spec:
        import numpy as np

        start, stop, num = spec["linspace"]
        return np.linspace(float(start), float(stop), int(num)).tolist()
    return [spec]


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = read_json(path)
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def expand_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config.get("grid", {})
    fixed = dict(config.get("fixed", {}))
    keys = list(grid)
    values: list[Iterable[Any]] = [_values(grid[k]) for k in keys]
    tasks: list[dict[str, Any]] = []
    for combo in itertools.product(*values) if keys else [()]:
        params = dict(fixed)
        params.update(dict(zip(keys, combo, strict=True)))
        task = {
            "suite": config["name"],
            "experiment": config["experiment"],
            "params": params,
            "output_root": config.get("output_root", f"outputs/{config['name']}"),
            "threads": int(config.get("threads", 1)),
            "config_version": config.get("version", 1),
        }
        task["task_id"] = stable_hash(task, length=20)
        tasks.append(task)
    expected = config.get("expected_tasks")
    if expected is not None and len(tasks) != int(expected):
        raise ValueError(
            f"Config {config['name']} expands to {len(tasks)} tasks; expected {expected}."
        )
    return tasks


def expand_paths(paths: list[str | Path]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for task in expand_config(load_config(path)):
            if task["task_id"] in seen:
                raise ValueError(f"Duplicate task id: {task['task_id']}")
            seen.add(task["task_id"])
            tasks.append(task)
    return tasks
