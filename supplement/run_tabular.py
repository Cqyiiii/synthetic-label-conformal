"""Paired 60-task candidate-selection supplement with resumable artifacts."""
from __future__ import annotations

import argparse
import concurrent.futures
import time
import traceback
from dataclasses import asdict

import common
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common import ROOT, PROJECT, CONFIG, write_json, utcnow, environment
from postconformal_selection import selection_scores, bounded_fourier_features, conformal_offset, pinball, quadratic_moment_estimate
from training import fit_candidates
from synrcp.v11 import _allocate_v11_roles, _prefix, _load_v11_data, _fit_teacher, _teacher_predict, _score, _threshold_bound
from synrcp.v11_neural import fit_mean_network
from synrcp.real_data import _column_medians, _impute
from synrcp.metrics import grouped_coverage_metrics, select_worst_slab, evaluate_slab, cross_fitted_ert
from synrcp.conformal import box_volume_summary


def allocate_roles(n,seed):
    original=_allocate_v11_roles(n,seed+1000)
    cp=_prefix(original['conformal'],n,.01)
    pool=_prefix(original['synthetic_pool'],n,.30)
    mid=len(cp)//2
    roles={k:original[k] for k in ['preprocessing','teacher_train','teacher_val','partition','slice_select','evaluation']}
    roles.update(base_train=_prefix(original['base_train'],n,.02),learning=cp[:mid],calibration=cp[mid:],
                 development=original['conformal'][len(cp):],pool_train=pool,pool_validation=original['synthetic_pool'][len(pool):])
    all_ids=np.concatenate(list(roles.values()))
    if len(np.unique(all_ids))!=len(all_ids):
        raise ValueError('Role overlap')
    return roles


def metric_row(q,s,roles,x,scale,phi_eval,tau,seed):
    gamma=conformal_offset(s['calibration']-q['calibration'],tau)
    radius=q['evaluation']+gamma
    cover=s['evaluation']<=radius
    cover_slice=s['slice_select']<=q['slice_select']+gamma
    row={'gamma':gamma,'coverage':float(cover.mean()),'coverage_error':float(abs(cover.mean()-tau)),
         'raw_pinball_test':float(pinball(s['evaluation']-q['evaluation'],tau).mean()),
         'final_pinball_test':float(pinball(s['evaluation']-radius,tau).mean()),
         'moment_discrepancy':quadratic_moment_estimate(cover.astype(float)-tau,phi_eval)}
    for k in [10,30]:
        met=grouped_coverage_metrics(x[roles['partition']],x[roles['evaluation']],cover,tau,k,seed+4000+k)
        row.update({f'group_{key}_k{k}':value for key,value in asdict(met).items()})
    slab=select_worst_slab(x[roles['slice_select']],cover_slice,min_mass=.1,directions=64,quantile_grid=7,seed=seed+4303)
    row.update(evaluate_slab(x[roles['evaluation']],cover,slab))
    row.update(cross_fitted_ert(x[roles['evaluation']],cover,tau,folds=5,seed=seed+4505))
    row.update(box_volume_summary(q['evaluation'],gamma,scale))
    return row


def execute_task(dataset,seed):
    threadpool_limits(1)
    dest=ROOT/'outputs/tabular'/f'{dataset}_seed{seed:02d}'
    if (dest/'result.json').exists():
        return {'dataset':dataset,'seed':seed,'status':'cached'}
    dest.mkdir(parents=True,exist_ok=True)
    started=utcnow(); tic=time.monotonic()
    write_json(dest/'started.json',{'dataset':dataset,'seed':seed,'started_at':started})
    try:
        raw_x,raw_y,meta=_load_v11_data(dataset,PROJECT/'legacy/Datasets',dataset_sample_seed=2026)
        roles=allocate_roles(len(raw_x),seed)
        np.savez_compressed(dest/'roles.npz',**roles)
        prep=roles['preprocessing']
        xi=_impute(raw_x,_column_medians(raw_x[prep]))
        yi_prep=_impute(raw_y[prep],_column_medians(raw_y[prep]))
        xs=StandardScaler().fit(xi[prep]); ys=StandardScaler().fit(yi_prep)
        x=xs.transform(xi)
        ymed=_column_medians(raw_y[prep])
        # Only named trusted roles receive true outcomes. No pool label arrays are created.
        def trusted_y(role):
            if role in ['pool_train','pool_validation']:
                raise RuntimeError('Synthetic-pool true outcomes are inaccessible')
            return ys.transform(_impute(raw_y[roles[role]],ymed))
        teacher=_fit_teacher(x[roles['teacher_train']],trusted_y('teacher_train'),strength='moderate',seed=seed+2000)
        mean=fit_mean_network(x[roles['base_train']],trusted_y('base_train'),seed=seed+3000,hidden=128,epochs=100,learning_rate=.002)
        if not mean.diagnostics['finite']:
            raise FloatingPointError('Base model nonfinite')
        mean_pred={r:mean.predict(x[idx]) for r,idx in roles.items() if r not in ['teacher_train','teacher_val','preprocessing','base_train']}
        scores={r:_score(trusted_y(r),mean_pred[r]) for r in ['learning','development']}
        synthetic={r:_score(_teacher_predict(teacher,x[roles[r]]),mean_pred[r]) for r in ['learning','development','pool_train','pool_validation']}
        bound=_threshold_bound(scores['learning'])
        kwargs={'seed':seed+3202,'tau':.9,'hidden':128,'learning_rate':.002,'bound':bound}
        bank=fit_candidates(x[roles['learning']],scores['learning'],x[roles['pool_train']],synthetic['pool_train'],synthetic['learning'],powers=CONFIG['powers'],checkpoints=CONFIG['checkpoints'],**kwargs)
        pred_roles=['development','pool_validation','calibration','evaluation','slice_select']
        predictions=[{r:c['predict'](x[roles[r]]) for r in pred_roles} for c in bank]
        if not all(np.isfinite(a).all() for p in predictions for a in p.values()):
            raise FloatingPointError('Nonfinite candidate predictions')
        frequencies=np.random.default_rng(seed+5000).normal(size=(x.shape[1],16))/np.sqrt(x.shape[1])
        phi={r:bounded_fourier_features(x[roles[r]],frequencies) for r in ['development','pool_validation','evaluation']}
        q_dev=np.stack([p['development'] for p in predictions]); q_pool=np.stack([p['pool_validation'] for p in predictions])
        selection=selection_scores(q_dev,scores['development'],phi['development'],tau=.9,seed=seed+6000,
                                   sdag_dev=synthetic['development'],q_pool=q_pool,sdag_pool=synthetic['pool_validation'],phi_pool=phi['pool_validation'])
        selection['candidate_ids']=[c['id'] for c in bank]
        selection['selected_ids']={k:bank[v]['id'] for k,v in selection['selected'].items()}
        selection['frozen_at']=utcnow()
        write_json(dest/'selection.json',selection)
        # Label-cost control never participates in candidate selection.
        extra_idx=np.r_[roles['learning'],roles['development']]
        extra=fit_candidates(x[extra_idx],np.r_[scores['learning'],scores['development']],x[roles['pool_train']],synthetic['pool_train'],np.zeros(len(extra_idx)),powers=[0.],checkpoints=[100],**kwargs)[0]
        extra['id']='extra_supervised_epoch100'; extra['resource_control']=True
        bank.append(extra)
        predictions.append({r:extra['predict'](x[roles[r]]) for r in pred_roles})
        # Selection is already saved; final outcomes are first transformed here.
        for r in ['calibration','evaluation','slice_select']:
            scores[r]=_score(trusted_y(r),mean_pred[r])
        final_opened_at=utcnow()
        import torch
        torch.save({c['id']:c['state'] for c in bank},dest/'checkpoints.pt')
        cache={f'q_{r}':np.stack([p[r] for p in predictions]) for r in pred_roles}
        cache.update({f's_{r}':s for r,s in scores.items()})
        cache.update(sdag_development=synthetic['development'],sdag_pool_validation=synthetic['pool_validation'],frequencies=frequencies,
                     phi_development=phi['development'],phi_pool_validation=phi['pool_validation'],phi_evaluation=phi['evaluation'])
        np.savez_compressed(dest/'predictions.npz',**cache)
        invariants=[]
        for shift in CONFIG['global_shift_values']:
            shifted=selection_scores(q_dev+shift,scores['development'],phi['development'],tau=.9,seed=seed+6000,
                                     sdag_dev=synthetic['development'],q_pool=q_pool+shift,sdag_pool=synthetic['pool_validation'],phi_pool=phi['pool_validation'])
            diffs={k:float(np.max(np.abs(np.array(selection['scores'][k])-shifted['scores'][k]))) for k in ['calibrated_pinball','coverage_beta0','coverage_beta1']}
            final_delta=0.
            for p in predictions[:6]:
                g=conformal_offset(scores['calibration']-p['calibration'],.9)
                gs=conformal_offset(scores['calibration']-p['calibration']-shift,.9)
                final_delta=max(final_delta,float(np.max(np.abs(p['evaluation']+g-(p['evaluation']+shift+gs)))))
            invariants.append({'shift':shift,**diffs,'final_threshold_max_error':final_delta,
                               'raw_score_max_change':float(np.max(np.abs(np.array(selection['scores']['raw_pinball'])-shifted['scores']['raw_pinball'])))})
            if max(*diffs.values(),final_delta)>1e-10:
                raise AssertionError('Global-shift invariant failed')
        evaluated=[]
        for c,p in zip(bank,predictions):
            evaluated.append({'dataset':dataset,'seed':seed,'candidate':c['id'],'power':c['power'],'epoch':c['epoch'],
                              **metric_row(p,scores,roles,x,ys.scale_,phi['evaluation'],.9,seed)})
        by_id={r['candidate']:r for r in evaluated}
        mapping={'Fixed-Supervised':'lambda0_epoch100','Fixed-PPI':'lambda1_epoch100','Extra-Supervised':extra['id'],**selection['selected_ids']}
        methods=[{'method':method,**by_id[cid]} for method,cid in mapping.items()]
        headroom={metric:{'bank_min':float(min(r[metric] for r in evaluated[:6])),
                          'bank_max':float(max(r[metric] for r in evaluated[:6])),
                          'fixed_ppi':by_id['lambda1_epoch100'][metric]} for metric in ['group_raw_k30','group_cross_product_k10','wsc','ert_l1_gain','moment_discrepancy']}
        result={'dataset':dataset,'seed':seed,'status':'complete','started_at':started,'completed_at':utcnow(),'elapsed_seconds':time.monotonic()-tic,
                'source':meta,'dimensions':{'x':x.shape[1],'y':raw_y.shape[1]},'role_counts':{k:len(v) for k,v in roles.items()},
                'quality':{'nonfinite_x':int((~np.isfinite(raw_x)).sum()),'nonfinite_y':int((~np.isfinite(raw_y)).sum()),'role_overlap':0,
                           'pool_true_outcome_access':0,'selection_frozen_at':selection['frozen_at'],'final_outcomes_opened_at':final_opened_at},
                'base_diagnostics':mean.diagnostics,'candidate_diagnostics':{c['id']:c['diagnostics'] for c in bank},
                'selection':selection,'invariants':invariants,'candidate_metrics':evaluated,'methods':methods,'headroom':headroom}
        write_json(dest/'result.json',result)
        pd.DataFrame(methods).to_csv(dest/'methods.csv',index=False)
        return {'dataset':dataset,'seed':seed,'status':'complete','seconds':round(time.monotonic()-tic,1)}
    except Exception:
        write_json(dest/f'failure_{time.time_ns()}.json',{'dataset':dataset,'seed':seed,'time':utcnow(),'traceback':traceback.format_exc()})
        raise


def main():
    p=argparse.ArgumentParser(); p.add_argument('--workers',type=int,default=3); p.add_argument('--smoke',action='store_true'); args=p.parse_args()
    write_json(ROOT/'outputs/environment.json',environment())
    tasks=[(d,s) for d in CONFIG['datasets'] for s in CONFIG['seeds']]
    if args.smoke:
        tasks=[('bike',0)]
    failures=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures={ex.submit(execute_task,*t):t for t in tasks}
        for f in concurrent.futures.as_completed(futures):
            task=futures[f]
            try:
                print(f.result(),flush=True)
            except Exception as error:
                failures.append({'dataset':task[0],'seed':task[1],'error':str(error)})
                print({'failed':task,'error':str(error)},flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} tasks failed; see per-task failure logs')


if __name__=='__main__':
    main()
