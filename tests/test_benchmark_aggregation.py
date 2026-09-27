"""Regression coverage for preserving benchmark rows and experimental conditions."""
import json

import pandas as pd

from synrcp.aggregate import aggregate_output


def test_v11_results_are_exported_without_merging_experimental_conditions(tmp_path):
    tasks = tmp_path / 'tasks'
    tasks.mkdir()
    for index, (teacher, budget) in enumerate([('weak', 'low'), ('strong', 'high')]):
        payload = {
            'task': {'task_id': str(index), 'suite': 'v11_teacher_quality',
                     'experiment': 'v11_teacher_quality', 'params': {'synthetic_prop': 0.3}},
            'result': {'dataset': 'example', 'seed': 0,
                       'condition': {'teacher_strength': teacher, 'label_budget': budget},
                       'methods': [
                           {'method': name, 'coverage_error': error,
                            'supervised_counterpart': 'RCP', 'method_family': 'RCP',
                            'legacy_alias': name, 'synthetic_use': 'none' if name == 'RCP' else 'PPI',
                            'grouped': [{'k': 30, 'raw': error}]}
                           for name, error in [('RCP', 0.03), ('RCP-PPI', 0.02)]]}
        }
        (tasks / f'{index}.json').write_text(json.dumps(payload))
    outputs = aggregate_output(tmp_path)
    rows = pd.read_csv(outputs['rows'])
    assert len(rows) == 4
    assert rows.groupby('task_id')['method'].nunique().to_dict() == {0: 2, 1: 2}
    assert set(zip(rows.teacher_strength, rows.label_budget)) == {('weak', 'low'), ('strong', 'high')}
    comparisons = pd.read_csv(outputs['comparisons'])
    assert comparisons.paired_n.eq(1).all()
    assert set(zip(comparisons.teacher_strength, comparisons.label_budget)) == {('weak', 'low'), ('strong', 'high')}
