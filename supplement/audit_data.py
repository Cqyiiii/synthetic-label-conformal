"""Dataset quality receipt; preserve original row handling without new filtering."""
import common
import numpy as np
import pandas as pd
from common import ROOT, PROJECT, CONFIG, write_json
from synrcp.v11 import _load_v11_data

rows=[]; overlaps=[]
for dataset in CONFIG['datasets']:
    x,y,meta=_load_v11_data(dataset,PROJECT/'legacy/Datasets',dataset_sample_seed=2026)
    frame=pd.DataFrame(np.c_[x,y])
    codes,_=pd.factorize(pd.MultiIndex.from_frame(frame))
    for seed in CONFIG['seeds']:
        roles=np.load(ROOT/'outputs/tabular'/f'{dataset}_seed{seed:02d}'/'roles.npz')
        fitted=np.concatenate([roles[k] for k in ['preprocessing','teacher_train','base_train','learning']])
        fit_codes=set(codes[fitted])
        overlaps.append({'dataset':dataset,'seed':seed,
                         'evaluation_rows_equal_to_any_fit_record':sum(int(codes[i] in fit_codes) for i in roles['evaluation']),
                         'calibration_rows_equal_to_any_fit_record':sum(int(codes[i] in fit_codes) for i in roles['calibration']),
                         'development_rows_equal_to_any_fit_record':sum(int(codes[i] in fit_codes) for i in roles['development'])})
    rows.append({'dataset':dataset,**meta,'x_columns':x.shape[1],'y_columns':y.shape[1],
                 'exact_joint_duplicate_rows':int(frame.duplicated().sum()),
                 'duplicate_x_rows':int(pd.DataFrame(x).duplicated().sum()),
                 'nonfinite_x':int((~np.isfinite(x)).sum()),'nonfinite_y':int((~np.isfinite(y)).sum()),
                 'new_exclusions':0,'handling':'V18 loader unchanged; SGEMM fixed 50k cap; role-specific preprocessing'})
pd.DataFrame(rows).to_csv(ROOT/'outputs/analysis/data_quality.csv',index=False)
write_json(ROOT/'outputs/analysis/data_quality.json',rows)
pd.DataFrame(overlaps).to_csv(ROOT/'outputs/analysis/duplicate_overlap.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
