from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[_key] = '1'

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
LEGACY = PROJECT/'src'
sys.path.insert(0,str(LEGACY))
CONFIG = json.loads((ROOT/'config.json').read_text())


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    temp.replace(path)


def environment():
    import importlib.metadata as metadata
    return {'python':sys.version.split()[0], 'interpreter':'active Python environment',
            'versions':{p:metadata.version(p) for p in ['numpy','scipy','pandas','torch','scikit-learn','matplotlib','pytest','pyarrow']},
            'device':'cpu','threads_per_process':1,'llm_api_calls':0,'new_embedding_inference':0}
