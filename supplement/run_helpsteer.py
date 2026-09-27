"""Reconstruct three published HelpSteer2 arms, then audit profiled risk."""
from __future__ import annotations

import copy
import importlib.util
import time
from pathlib import Path

import common
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from common import ROOT, PROJECT, CONFIG, write_json, utcnow
from postconformal_selection import conformal_offset, pinball

LEGACY=PROJECT/'llm/helpsteer'
_spec=importlib.util.spec_from_file_location('helpsteer_original',LEGACY/'run_downstream_pilot.py')
old=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(old)


def split_groups(keys,seed):
    unique=np.unique(keys).copy()
    np.random.default_rng(seed).shuffle(unique)
    left=np.isin(keys,unique[:len(unique)//2])
    return np.flatnonzero(left),np.flatnonzero(~left)


def cross_calibrated_risk(q,s,folds):
    values=[]; offsets=[]
    for i in range(2):
        cal,val=folds[i],folds[1-i]
        gamma=conformal_offset(s[cal]-q[cal],.9)
        offsets.append(gamma)
        values.append(float(pinball(s[val]-q[val]-gamma,.9).mean()))
    return float(np.mean(values)), offsets


def main():
    threadpool_limits(1); torch.set_num_threads(1)
    dest=ROOT/'outputs/helpsteer'; dest.mkdir(parents=True,exist_ok=True)
    tic=time.monotonic(); started=utcnow()
    data=pd.read_parquet(LEGACY/'data/helpsteer2.parquet')
    pred=pd.read_parquet(LEGACY/'results/pilot_predictions.parquet')
    syncols=[f'syn_{a}_{k}' for a in ['flash','pro'] for k in old.LABELS]
    data=data.merge(pred[['row_id',*syncols]],on='row_id',how='left',validate='one_to_one')
    emb=np.load(LEGACY/'data/helpsteer2_embeddings.npz')
    if len(set(emb['row_ids']))!=len(emb['row_ids']):
        raise ValueError('Duplicate embeddings')
    lookup={str(i):x.astype(np.float32) for i,x in zip(emb['row_ids'],emb['features'])}
    frame={r:data.loc[data.role.eq(r)].sort_values('row_id').copy() for r in ['base_train','pilot','conformal','test']}
    x={r:np.stack([lookup[str(i)] for i in f.row_id]) for r,f in frame.items()}
    y={r:f[old.LABELS].to_numpy(np.float32)/4 for r,f in frame.items()}
    role_keys={r:set(f.prompt_key) for r,f in frame.items()}
    overlap={f'{a}:{b}':len(role_keys[a]&role_keys[b]) for i,a in enumerate(frame) for b in list(frame)[i+1:]}
    if max(overlap.values())!=0:
        raise ValueError('Prompt overlap across outer roles')
    cache=dest/'base_predictions.npz'
    if cache.exists():
        mu=dict(np.load(cache))
    else:
        base=old.fit_mean(x['base_train'],y['base_train'],350,'cpu')
        torch.save(base.state_dict(),dest/'base_model.pt')
        mu={r:old.predict(base,v) for r,v in x.items()}
        np.savez_compressed(cache,**mu)
    scores={r:np.max(np.abs(y[r]-mu[r]),axis=1) for r in frame}
    syn={a:np.max(np.abs(frame['pilot'][[f'syn_{a}_{k}' for k in old.LABELS]].to_numpy(np.float32)/4-mu['pilot']),axis=1) for a in ['flash','pro']}
    if not all(np.isfinite(v).all() for v in syn.values()):
        raise ValueError('Missing synthetic scores')
    # Exact original prompt-separated development/final split.
    keys=frame['conformal'].prompt_key.drop_duplicates().to_numpy().copy()
    np.random.default_rng(20260804).shuffle(keys)
    sizes=frame['conformal'].groupby('prompt_key').size(); chosen=[]; count=0
    for key in keys:
        if count>=100:
            break
        chosen.append(str(key)); count+=int(sizes.loc[key])
    devmask=frame['conformal'].prompt_key.isin(chosen).to_numpy()
    dev=np.flatnonzero(devmask); cal=np.flatnonzero(~devmask)
    # Original K10 diagnostic groups use only frozen test X.
    groups=KMeans(n_clusters=10,n_init=20,random_state=20260804).fit_predict(x['test'])
    test_folds=split_groups(frame['test'].prompt_key.to_numpy(),20260925)
    dev_folds=split_groups(frame['conformal'].iloc[dev].prompt_key.to_numpy(),20260926)
    np.savez_compressed(dest/'audit_roles.npz',development=dev,calibration=cal,test_fold0=test_folds[0],test_fold1=test_folds[1],groups=groups,
                        **{f'{r}_row_ids':f.row_id.to_numpy(dtype=str) for r,f in frame.items()})
    records=[]; dependence=[]
    for seed in range(20):
        seed_dest=dest/f'seed{seed:02d}'; seed_dest.mkdir(exist_ok=True)
        if (seed_dest/'result.json').exists():
            import json
            loaded=json.loads((seed_dest/'result.json').read_text()); records.extend(loaded['metrics']); dependence.append(loaded['dependence'])
            print(f'HelpSteer2 seed {seed}: cached',flush=True); continue
        perm=np.random.default_rng(1000+seed).permutation(len(frame['pilot']))
        pair,aux=perm[:100],perm[100:]
        prompt_pair=set(frame['pilot'].iloc[pair].prompt_key)
        prompt_aux=set(frame['pilot'].iloc[aux].prompt_key)
        dep={'seed':seed,'paired_pool_prompt_overlap':len(prompt_pair&prompt_aux)}; dependence.append(dep)
        old.seed_everything(91000+seed)
        initial=old.QuantileNet(x['pilot'].shape[1]); state=copy.deepcopy(initial.state_dict())
        seed_rows=[]; predictions={}; states={}
        for name,kind,alias in [('RCP','rcp',None),('Flash-PPI','ppi','flash'),('Pro-PPI','ppi','pro')]:
            sy=np.zeros_like(scores['pilot']) if alias is None else syn[alias]
            aux_use=np.array([],dtype=int) if alias is None else aux
            model,epoch=old.fit_quantile(state,x['pilot'][pair],scores['pilot'][pair],sy[pair],x['pilot'][aux_use],sy[aux_use],kind,.9,260,x['conformal'][dev],scores['conformal'][dev])
            qtest=old.predict(model,x['test']); qconf=old.predict(model,x['conformal']); qdev=qconf[dev]
            vals=old.evaluate(qtest,qconf[cal],scores['test'],scores['conformal'][cal],mu['test'],groups,.9)
            cross_risk,offsets=cross_calibrated_risk(qtest.astype(float),scores['test'].astype(float),test_folds)
            dev_cross,dev_offsets=cross_calibrated_risk(qdev.astype(float),scores['conformal'][dev].astype(float),dev_folds)
            radius=np.clip(qtest+vals['gamma'],0,1); covered=scores['test']<=radius
            weighted=sum(np.mean(groups==g)*(covered[groups==g].mean()-.9)**2 for g in np.unique(groups))
            row={'seed':seed,'method':name,'selected_epoch':epoch,**vals,'cross_calibrated_pinball_test':cross_risk,
                 'raw_pinball_dev_adaptive':float(pinball(scores['conformal'][dev]-qdev,.9).mean()),
                 'cross_calibrated_pinball_dev_adaptive':dev_cross,'weighted_msce':float(weighted),
                 'test_offset_fold0':offsets[0],'test_offset_fold1':offsets[1],
                 'dev_offset_fold0':dev_offsets[0],'dev_offset_fold1':dev_offsets[1],
                 'final_pinball_test':float(pinball(scores['test']-qtest-vals['gamma'],.9).mean())}
            seed_rows.append(row); records.append(row)
            predictions[f'{name}_test']=qtest; predictions[f'{name}_conformal']=qconf
            states[name]=copy.deepcopy(model.state_dict())
        np.savez_compressed(seed_dest/'predictions.npz',**predictions,pair=pair,aux=aux,score_test=scores['test'],score_conformal=scores['conformal'])
        torch.save(states,seed_dest/'checkpoints.pt')
        write_json(seed_dest/'result.json',{'seed':seed,'metrics':seed_rows,'dependence':dep,'completed_at':utcnow()})
        print(f'HelpSteer2 seed {seed}: 3 arms complete',flush=True)
    pd.DataFrame(records).to_csv(dest/'metrics.csv',index=False)
    archived=pd.read_csv(LEGACY/'results/downstream_cost_matched/raw_metrics.csv')
    compared=pd.DataFrame(records).merge(archived,on=['seed','method'],suffixes=('_new','_old'),validate='one_to_one')
    compare={k:float((compared[f'{k}_new']-compared[f'{k}_old']).abs().max()) for k in ['true_pinball','coverage','msce','selected_epoch']}
    write_json(dest/'execution.json',{'started_at':started,'completed_at':utcnow(),'elapsed_seconds':time.monotonic()-tic,
                                    'rows':len(records),'seeds':20,'methods':['RCP','Flash-PPI','Pro-PPI'],'new_llm_calls':0,
                                    'role_counts':{r:len(f) for r,f in frame.items()},'development':len(dev),'calibration':len(cal),
                                    'test_fold_counts':[len(v) for v in test_folds],'test_prompt_counts':[len(set(frame['test'].iloc[v].prompt_key)) for v in test_folds],
                                    'outer_role_prompt_overlap':overlap,'within_pilot_dependence':dependence,
                                    'base_mae':float(np.abs(mu['base_train']-y['base_train']).mean()),'test_mae':float(np.abs(mu['test']-y['test']).mean()),
                                    'archived_max_absolute_difference':compare})


if __name__=='__main__':
    main()
