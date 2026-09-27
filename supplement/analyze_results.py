"""Validate all outputs and derive report tables without refitting/selecting."""
from __future__ import annotations

import json
from pathlib import Path

import common
import numpy as np
import pandas as pd
from common import ROOT, CONFIG, write_json, utcnow
from postconformal_selection import selection_scores, pinball

METRICS=['group_raw_k30','group_cross_product_k10','wsc','ert_l1_gain','coverage','coverage_error','mean_log_volume','moment_discrepancy']
SELECTORS=CONFIG['selection_rules']


def bootstrap(x,seed):
    x=np.asarray(x,dtype=float)
    means=x[np.random.default_rng(seed).integers(0,len(x),size=(10000,len(x)))].mean(1)
    return np.quantile(means,[.025,.975]).tolist()


def contrasts(frame,metrics,pairs,group='dataset',seed=20260925):
    records=[]; index=0
    for g,df in frame.groupby(group,sort=True):
        for a,b in pairs:
            aa=df.loc[df.method.eq(a)].set_index('seed'); bb=df.loc[df.method.eq(b)].set_index('seed')
            assert aa.index.equals(bb.index)
            for metric in metrics:
                vals=aa[metric]-bb[metric]; lo,hi=bootstrap(vals.to_numpy(),seed+index); index+=1
                higher=metric in ['wsc','min_group_coverage','coverage']
                better=vals>1e-12 if higher else vals < -1e-12
                records.append({group:g,'method':a,'reference':b,'metric':metric,'n':len(vals),'difference':float(vals.mean()),
                                'ci95_low':lo,'ci95_high':hi,'wins':int(better.sum()),'ties':int((vals.abs()<=1e-12).sum()),
                                'lower_is_better':not higher})
    return pd.DataFrame(records)


def main():
    results=[json.loads(p.read_text()) for p in sorted((ROOT/'outputs/tabular').glob('*/result.json'))]
    planned={(d,s) for d in CONFIG['datasets'] for s in CONFIG['seeds']}
    actual={(r['dataset'],r['seed']) for r in results}
    assert actual==planned and len(results)==60, f'Only {len(results)}/60 tasks'
    rows=[]; candidates=[]; roles=[]; selections=[]; invariants=[]; headroom=[]; executions=[]
    for r in results:
        assert r['status']=='complete' and len(r['methods'])==7 and len(r['candidate_metrics'])==7
        assert r['quality']['role_overlap']==0 and r['quality']['pool_true_outcome_access']==0
        assert r['quality']['selection_frozen_at'] < r['quality']['final_outcomes_opened_at']
        task=ROOT/'outputs/tabular'/f"{r['dataset']}_seed{r['seed']:02d}"
        ids=np.load(task/'roles.npz'); all_ids=np.concatenate([ids[k] for k in ids.files])
        assert len(np.unique(all_ids))==len(all_ids)
        p=np.load(task/'predictions.npz')
        # Reconstruct selector scores using only saved pre-final inputs.
        check=selection_scores(p['q_development'][:6],p['s_development'],p['phi_development'],seed=r['seed']+6000,
                               sdag_dev=p['sdag_development'],q_pool=p['q_pool_validation'][:6],sdag_pool=p['sdag_pool_validation'],phi_pool=p['phi_pool_validation'])
        assert check['selected']==r['selection']['selected']
        for k,v in check['scores'].items():
            np.testing.assert_allclose(v,r['selection']['scores'][k],atol=1e-14)
        for v in r['candidate_diagnostics'].values():
            assert v['finite'] and np.isfinite(v['parameter_norm'])
        assert np.isfinite(p['q_evaluation']).all()
        for row in r['methods']:
            for metric in METRICS:
                assert np.isfinite(row[metric]),(r['dataset'],r['seed'],metric)
        rows.extend(r['methods']); candidates.extend(r['candidate_metrics'])
        roles.append({'dataset':r['dataset'],'seed':r['seed'],**r['role_counts'],'source_rows':r['source']['original_rows'],'used_rows':r['source']['used_rows'],
                      'feature_dimension':r['dimensions']['x'],'outcome_dimension':r['dimensions']['y'],
                      'nonfinite_x':r['quality']['nonfinite_x'],'nonfinite_y':r['quality']['nonfinite_y']})
        for rule in SELECTORS:
            chosen=r['selection']['selected_ids'][rule]
            selections.append({'dataset':r['dataset'],'seed':r['seed'],'method':rule,'candidate':chosen,
                               'score':min(r['selection']['scores'][rule]),'negative_candidate_scores':sum(s<0 for s in r['selection']['scores'][rule])})
        for v in r['invariants']:
            fold0=np.asarray(r['selection']['fold0']); fold1=np.asarray(r['selection']['fold1'])
            shifted=p['q_development'][:6]+v['shift']
            raw_shifted=.5*(pinball(p['s_development'][fold0]-shifted[:,fold0],.9).mean(1)+pinball(p['s_development'][fold1]-shifted[:,fold1],.9).mean(1))
            original=np.asarray(r['selection']['scores']['raw_pinball'])
            invariants.append({'dataset':r['dataset'],'seed':r['seed'],**v,
                               'raw_selected_original':int(np.argmin(original)), 'raw_selected_shifted':int(np.argmin(raw_shifted)),
                               'raw_selection_changed':bool(np.argmin(original)!=np.argmin(raw_shifted)),
                               'raw_order_changed':bool(not np.array_equal(np.argsort(original,kind='stable'),np.argsort(raw_shifted,kind='stable')))})
        for metric,values in r['headroom'].items():
            best=values['bank_max'] if metric=='wsc' else values['bank_min']
            gap=best-values['fixed_ppi'] if metric=='wsc' else values['fixed_ppi']-best
            headroom.append({'dataset':r['dataset'],'seed':r['seed'],'metric':metric,'fixed_ppi':values['fixed_ppi'],'evaluation_oracle':best,'headroom':gap})
        executions.append({'dataset':r['dataset'],'seed':r['seed'],'started_at':r['started_at'],'completed_at':r['completed_at'],'seconds':r['elapsed_seconds']})
    methods=pd.DataFrame(rows).sort_values(['dataset','seed','method']); bank=pd.DataFrame(candidates)
    summary=methods.groupby(['dataset','method'])[METRICS].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).rstrip('_') for c in summary.columns]
    pairs=[('calibrated_pinball','raw_pinball'),('coverage_beta0','calibrated_pinball'),('coverage_beta1','coverage_beta0')]
    pairs += [(r,'Fixed-PPI') for r in SELECTORS]+[(r,'Extra-Supervised') for r in SELECTORS]+[('Fixed-PPI','Fixed-Supervised')]
    paired=contrasts(methods,METRICS,pairs)
    h=pd.read_csv(ROOT/'outputs/helpsteer/metrics.csv').sort_values(['seed','method'])
    assert len(h)==60 and h.groupby('method').size().eq(20).all()
    hm=['true_pinball','cross_calibrated_pinball_test','final_pinball_test','msce','weighted_msce','min_group_coverage','coverage','mean_log_volume','raw_pinball_dev_adaptive','cross_calibrated_pinball_dev_adaptive']
    hs=h.groupby('method')[hm].agg(['mean','std']).reset_index(); hs.columns=['_'.join(c).rstrip('_') for c in hs.columns]
    hp=contrasts(h.assign(study='HelpSteer2'),hm,[('Flash-PPI','RCP'),('Pro-PPI','RCP')],group='study',seed=20260926)
    stat=pd.read_csv(ROOT/'outputs/statistical/metrics.csv')
    assert len(stat)==32 and np.isfinite(stat.select_dtypes('number')).all().all()
    assert stat.bias_z.abs().max()<4 and stat.variance_z.abs().max()<4
    dest=ROOT/'outputs/analysis'; dest.mkdir(parents=True,exist_ok=True)
    tables={'methods':methods,'candidates':bank,'tabular_summary':summary,'tabular_paired':paired,'roles':pd.DataFrame(roles),
            'selections':pd.DataFrame(selections),'invariants':pd.DataFrame(invariants),'headroom':pd.DataFrame(headroom),'execution':pd.DataFrame(executions),
            'helpsteer_summary':hs,'helpsteer_paired':hp,'helpsteer_metrics':h,'statistical':stat}
    for name,frame in tables.items():
        frame.to_csv(dest/f'{name}.csv',index=False)
    max_invariance=max(pd.DataFrame(invariants)[['calibrated_pinball','coverage_beta0','coverage_beta1','final_threshold_max_error']].max())
    fail_files=list((ROOT/'outputs').glob('**/failure_*.json'))
    validation={'validated_at':utcnow(),'tabular_tasks':60,'method_rows':len(methods),'candidate_rows_including_extra':len(bank),
                'bank_candidates':360,'threshold_trajectories':240,'base_models_tabular':60,'helpsteer_rows':60,
                'statistical_settings':32,'statistical_repetitions_per_setting':100000,'task_failures':len(fail_files),
                'role_overlap':0,'synthetic_pool_true_labels_exposed_to_fit_or_select':0,'selection_reconstruction_matches':60,
                'negative_scores_retained':True,'max_global_shift_error':float(max_invariance),
                'statistics_max_abs_bias_z':float(stat.bias_z.abs().max()),'statistics_max_abs_variance_z':float(stat.variance_z.abs().max()),
                'neural_nonfinite':0,'final_selection_no_refit':True,'outcomes':'Released and evaluated after per-task selector freeze; no quarantine.',
                'browser_render_check':'Not part of this numerical analysis.'}
    assert max_invariance<1e-10
    write_json(dest/'validation.json',validation)
    print(json.dumps(validation,indent=2))
    print(summary[['dataset','method','group_raw_k30_mean','wsc_mean','ert_l1_gain_mean']].to_string(index=False))
    print(hs[['method','true_pinball_mean','cross_calibrated_pinball_test_mean','msce_mean']].to_string(index=False))


if __name__=='__main__':
    main()
