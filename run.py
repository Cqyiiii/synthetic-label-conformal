#!/usr/bin/env python3
"""Portable entry points. Paths resolve relative to this extracted package."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key,'1')

BENCHMARK = {name:name for name in ['main','teacher_quality','pool_size','label_budget','ablations']}
SUPPLEMENT = {'tabular':'run_tabular.py','helpsteer':'run_helpsteer.py','statistics':'run_statistics.py',
              'analyze':'analyze_results.py','audit':'audit_data.py','figures':'make_figures.py'}


def configured_tasks(family, suite='all'):
    from synrcp.config import expand_config, load_config
    paths=sorted((ROOT/'configs'/family).glob('*.json'))
    if suite!='all':
        target=f'v11_{suite}' if family=='benchmark' else suite
        paths=[p for p in paths if p.stem==target]
    if not paths:
        raise ValueError(f'No configuration for {family}/{suite}')
    return [task for path in paths for task in expand_config(load_config(path))]


def child(command, cwd=ROOT):
    env=dict(os.environ)
    env['PYTHONPATH']=str(ROOT/'src')
    subprocess.run([sys.executable,*map(str,command)],cwd=cwd,env=env,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('plan',help='Inspect every frozen task count; does not train or download.')
    sub.add_parser('check-results',help='Validate included numeric reference results.')
    for family in ['benchmark','theory']:
        p=sub.add_parser(family)
        p.add_argument('--suite',default='all',choices=['all',*BENCHMARK] if family=='benchmark' else ['all','conformal_null','mechanism_2x2','paired_score_oracle','signed_power','calibration_density'])
        p.add_argument('--smoke',action='store_true',help='Run the first task only, in a separate smoke output directory.')
    sub.add_parser('followup',help='Run the derivative audit and full 270-task resource scaling study.')
    p=sub.add_parser('helpsteer',help='Run the original cached-label downstream study; makes no API request.')
    p.add_argument('--smoke',action='store_true',help='One seed with the original epoch budgets.')
    p=sub.add_parser('supplement')
    p.add_argument('--part',choices=list(SUPPLEMENT),required=True)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--smoke',action='store_true',help='Supported for tabular only: one Bike task.')
    args=parser.parse_args()
    if args.command=='plan':
        from synrcp.followup import build_scaling_plan
        cfg=json.loads((ROOT/'supplement/config.json').read_text())
        print(json.dumps({'benchmark_tasks':len(configured_tasks('benchmark')),
                          'main_tasks':len(configured_tasks('benchmark','main')),
                          'theory_tasks':len(configured_tasks('theory')),
                          'followup_tasks':len(build_scaling_plan()),
                          'supplement_tabular_tasks':len(cfg['datasets'])*len(cfg['seeds']),
                          'benchmark_suite_tasks':{s:len(configured_tasks('benchmark',s)) for s in BENCHMARK}},indent=2))
    elif args.command=='check-results':
        child([ROOT/'tools/validate_reference.py'])
    elif args.command in ['benchmark','theory']:
        from synrcp.runner import execute_task
        from synrcp.aggregate import aggregate_output
        tasks=configured_tasks(args.command,args.suite)
        if args.smoke:
            tasks=tasks[:1]
            tasks[0]['output_root']=f'outputs/smoke_{args.command}'
        output_roots=set()
        for i,task in enumerate(tasks,1):
            execute_task(task,ROOT)
            output_roots.add(task['output_root'])
            print(f'{args.command}: {i}/{len(tasks)} completed',flush=True)
        for output in sorted(output_roots):
            aggregate_output(ROOT/output)
    elif args.command=='followup':
        child(['-m','synrcp.followup','--output-dir',ROOT/'outputs/theory_followup'])
    elif args.command=='helpsteer':
        output=ROOT/'outputs'/('smoke_helpsteer' if args.smoke else 'helpsteer')
        child(['run_downstream_pilot.py','--seeds','1' if args.smoke else '20',
               '--include-cost-match','--device','cpu','--output-dir',output],ROOT/'llm/helpsteer')
    elif args.command=='supplement':
        if args.smoke and args.part!='tabular':
            parser.error('--smoke is supported only for supplement --part tabular')
        command=[ROOT/'supplement'/SUPPLEMENT[args.part]]
        if args.part=='tabular':
            command += ['--workers',str(args.workers)]
            if args.smoke:
                command.append('--smoke')
        child(command)


if __name__=='__main__':
    main()
