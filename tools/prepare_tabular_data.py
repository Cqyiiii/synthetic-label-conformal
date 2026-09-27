"""Import the public benchmark tables and verify the expected raw schemas.

No model is fitted. Public source links and format details are in docs/DATA.md.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def prepare(source, destination):
    catalog=json.loads((ROOT/'docs/tabular_data.json').read_text())
    destination.mkdir(parents=True,exist_ok=True)
    prepared=[]
    for spec in catalog:
        filename=spec['file']; path=source/filename; target=destination/filename
        if path.exists():
            if path.resolve()!=target.resolve():
                shutil.copyfile(path,target)
        elif filename=='bike_hour.csv' and (source/'hour.csv').exists():
            shutil.copyfile(source/'hour.csv',target)
        elif filename=='super.csv' and (source/'train.csv').exists():
            shutil.copyfile(source/'train.csv',target)
        elif filename=='naval.csv' and (source/'data.txt').exists():
            pd.read_csv(source/'data.txt',sep=r'\s+',header=None).to_csv(target,index=False)
        elif filename=='gt_full.csv' and all((source/f'gt_{year}.csv').exists() for year in range(2011,2016)):
            pd.concat([pd.read_csv(source/f'gt_{year}.csv') for year in range(2011,2016)],ignore_index=True).to_csv(target,index=True)
        else:
            raise FileNotFoundError(f'Missing {filename}; see docs/DATA.md for the public source and conversion.')
        frame=pd.read_csv(target,sep='\t' if target.suffix=='.tsv' else ',')
        if len(frame)!=spec['rows'] or list(frame.columns)!=spec['columns']:
            raise ValueError(f'{filename}: row count or ordered columns differ from the recorded experiment input')
        prepared.append({'file':filename,'rows':len(frame),'columns':len(frame.columns)})
    return prepared


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-local',type=Path,required=True,help='Directory with the eight recorded files or documented public raw formats.')
    parser.add_argument('--destination',type=Path,default=ROOT/'legacy/Datasets')
    args=parser.parse_args()
    print(json.dumps(prepare(args.from_local,args.destination),indent=2))


if __name__=='__main__':
    main()
