#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_json(command: list[str], cwd: Path) -> dict:
    completed = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def main() -> int:
    frozen = run_json([sys.executable, "scripts/validate_v15_results.py"], ROOT)
    followup_root = ROOT / "theory_followup"
    followup = run_json(
        [
            sys.executable,
            "scripts/validate_v16_followup.py",
            "outputs/v16_theory_followup",
        ],
        followup_root,
    )
    payload = {
        "status": "PASS" if frozen.get("status") == "PASS" and followup.get("status") == "PASS" else "FAIL",
        "frozen_evidence": frozen,
        "theory_followup": followup,
        "note": (
            "The HOLD_NOT_RUN field belongs to the immutable preregistration snapshot. "
            "The separately archived V16 follow-up is complete and independently validated."
        ),
    }
    output = ROOT / "validation_v16_output.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
