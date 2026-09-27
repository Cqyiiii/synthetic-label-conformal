from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .aggregate import aggregate_output
from .config import expand_config, expand_paths, load_config
from .runner import execute_task, load_plan


def project_root() -> Path:
    override = os.environ.get("SYNRCP_PROJECT_ROOT")
    if override:
        return Path(override).resolve()
    return Path.cwd().resolve()


def command_plan(args: argparse.Namespace) -> None:
    tasks = expand_paths(args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, sort_keys=True) + "\n")
    print(f"Wrote {len(tasks)} tasks to {output}")


def command_run_task(args: argparse.Namespace) -> None:
    tasks = load_plan(args.plan)
    if args.index < 0 or args.index >= len(tasks):
        raise IndexError(f"Task index {args.index} is outside [0, {len(tasks)-1}].")
    path = execute_task(tasks[args.index], project_root(), force=args.force)
    print(path)


def command_run_config(args: argparse.Namespace) -> None:
    tasks = expand_config(load_config(args.config))
    for index, task in enumerate(tasks):
        path = execute_task(task, project_root(), force=args.force)
        print(f"[{index+1}/{len(tasks)}] {path}")


def command_slurm(args: argparse.Namespace) -> None:
    tasks = load_plan(args.plan)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    script = f"""#!/usr/bin/env bash
#SBATCH --job-name={args.job_name}
#SBATCH --array=0-{len(tasks)-1}
#SBATCH --time={args.time}
#SBATCH --mem={args.memory}
#SBATCH --cpus-per-task={args.cpus}
set -euo pipefail
cd \"${{SLURM_SUBMIT_DIR}}\"
export OMP_NUM_THREADS={args.cpus}
export MKL_NUM_THREADS={args.cpus}
synrcp run-task --plan {args.plan} --index \"${{SLURM_ARRAY_TASK_ID}}\"
"""
    output.write_text(script, encoding="utf-8")
    output.chmod(0o755)
    print(output)


def command_aggregate(args: argparse.Namespace) -> None:
    outputs = aggregate_output(args.output_root)
    for key, value in outputs.items():
        print(f"{key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synrcp")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Expand one or more JSON configs into a JSONL task plan.")
    plan.add_argument("--config", action="append", required=True)
    plan.add_argument("--output", required=True)
    plan.set_defaults(func=command_plan)

    run_task = sub.add_parser("run-task", help="Run one task from a JSONL plan.")
    run_task.add_argument("--plan", required=True)
    run_task.add_argument("--index", type=int, required=True)
    run_task.add_argument("--force", action="store_true")
    run_task.set_defaults(func=command_run_task)

    run_config = sub.add_parser("run-config", help="Run every task in one config sequentially.")
    run_config.add_argument("--config", required=True)
    run_config.add_argument("--force", action="store_true")
    run_config.set_defaults(func=command_run_config)

    slurm = sub.add_parser("slurm", help="Create a resumable SLURM array script.")
    slurm.add_argument("--plan", required=True)
    slurm.add_argument("--output", required=True)
    slurm.add_argument("--job-name", default="synrcp")
    slurm.add_argument("--time", default="24:00:00")
    slurm.add_argument("--memory", default="16G")
    slurm.add_argument("--cpus", type=int, default=4)
    slurm.set_defaults(func=command_slurm)

    aggregate = sub.add_parser("aggregate", help="Aggregate completed task JSON files.")
    aggregate.add_argument("--output-root", required=True)
    aggregate.set_defaults(func=command_aggregate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
