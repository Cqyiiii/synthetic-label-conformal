"""Check extracted report transformations against supplied numerical evidence."""
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_supplement_display_tables_match_reference(tmp_path):
    reference = ROOT / 'reference_results/supplement'
    result = subprocess.run([sys.executable, str(ROOT / 'supplement/display_tables.py'),
                             '--analysis-dir', str(reference), '--output-dir', str(tmp_path)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    names = ['main_display', 'resource_display', 'selection_frequencies',
             'headroom_display', 'invariance_display', 'variance_ratios']
    for name in names:
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / f'{name}.csv'),
                                      pd.read_csv(reference / f'{name}.csv'),
                                      check_exact=False, rtol=1e-12, atol=1e-12)
