from pathlib import Path

from synrcp_v11.plan import load_plan, validate_plan


def test_frozen_v11_plan_is_valid() -> None:
    path = Path(__file__).resolve().parents[1] / "plans" / "all_tasks.jsonl"
    rows = load_plan(path)
    assert len(rows) == 740
    assert validate_plan(rows) == []
