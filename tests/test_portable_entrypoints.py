"""These fail if launchers silently depend on the preparer's original checkout."""
import json
import subprocess
import sys
from pathlib import Path
import argparse
import importlib
import shlex
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_plan_from_unrelated_working_directory(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT/'run.py'), 'plan'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary['benchmark_tasks'] == 740
    assert summary['main_tasks'] == 240
    assert summary['followup_tasks'] == 270
    assert summary['supplement_tabular_tasks'] == 60


def test_supplement_imports_without_installed_or_original_checkout(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT/'supplement/run_tabular.py'), '--help'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--workers' in result.stdout


def test_helpsteer_reconstruction_imports_from_packaged_dependency(tmp_path):
    code = 'import sys; sys.path.insert(0, sys.argv[1]); import run_helpsteer; print(len(run_helpsteer.old.LABELS))'
    result = subprocess.run([sys.executable, '-c', code, str(ROOT/'supplement')],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '5'


def test_generated_slurm_command_has_an_installable_entry_point(tmp_path):
    from synrcp.cli import command_slurm
    plan = tmp_path / 'plan.jsonl'
    plan.write_text('{}\n')
    script = tmp_path / 'submit.sh'
    command_slurm(argparse.Namespace(plan=str(plan), output=str(script),
                                    job_name='review', time='00:10:00', memory='1G', cpus=1))
    executable = shlex.split(script.read_text().splitlines()[-1])[0]
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    entry = config['project'].get('scripts', {}).get(executable)
    assert entry, f'Generated scheduler command {executable!r} is not installed'
    module, function = entry.split(':')
    assert callable(getattr(importlib.import_module(module), function))
