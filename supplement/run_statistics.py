"""Finite-support exact variance and multinomial Monte Carlo experiment."""
from __future__ import annotations

import common
import numpy as np
import pandas as pd
from common import ROOT, CONFIG, write_json, utcnow


def exact_variance(mu,sa,sb,n,N,beta):
    first=float(4*mu@(sa/n+beta**2*sb/N)@mu)
    second_a=float(2*np.trace(sa@sa)/(n*(n-1)))
    second_b=float(2*beta**4*np.trace(sb@sb)/(N*(N-1)))
    cross=float(4*beta**2*np.trace(sa@sb)/(n*N))
    return {'first_order':first,'second_a':second_a,'second_b':second_b,'cross':cross,
            'total':first+second_a+second_b+cross}


def finite_support(p,ps,kind,tau=.9):
    rows,probs=[],[]
    for g in range(4):
        independent=p[g]*ps[g]
        joint={'positive':independent+.85*(min(p[g],ps[g])-independent),
               'independent':independent,'negative':max(0.,p[g]+ps[g]-1)}[kind]
        for true,syn,prob in [(1,1,joint),(1,0,p[g]-joint),(0,1,ps[g]-joint),(0,0,1-p[g]-ps[g]+joint)]:
            phi=np.eye(4)[g]*2
            rows.append((phi*(true-tau),phi*(syn-tau)))
            probs.append(max(0.,prob)/4)
    rows=np.asarray(rows)
    probs=np.asarray(probs); probs/=probs.sum()
    return rows[:,0,:],rows[:,1,:],probs


def run():
    started=utcnow(); R=CONFIG['statistical_repetitions']; rng=np.random.default_rng(20260925)
    output=[]; specs=[]
    for case,kind,p,ps in [
        ('nonzero_positive','positive',[.75,.85,.95,.99],[.77,.87,.93,.97]),
        ('nonzero_independent','independent',[.75,.85,.95,.99],[.77,.87,.93,.97]),
        ('nonzero_negative','negative',[.75,.85,.95,.99],[.77,.87,.93,.97]),
        ('zero_positive','positive',[.9]*4,[.88,.90,.91,.92]),
    ]:
        true,syn,prob=finite_support(np.array(p),np.array(ps),kind)
        specs.append({'case':case,'true_coverage':p,'synthetic_coverage':ps,'coupling':kind,'support_probabilities':prob.tolist()})
        for n in CONFIG['statistical_sample_sizes']:
            N=5*n
            ca=rng.multinomial(n,prob,size=R)
            cb=rng.multinomial(N,prob,size=R)
            for beta in [0.,1.]:
                a=true-beta*syn; b=syn
                ma,mb=prob@a,prob@b
                sa=(a-ma).T@((a-ma)*prob[:,None]); sb=(b-mb).T@((b-mb)*prob[:,None])
                mu=ma+beta*mb; truth=float(mu@mu)
                mean_a=ca@a/n; mean_b=cb@b/N
                noise_a=((ca@(a*a).sum(1))-n*(mean_a*mean_a).sum(1))/(n*(n-1))
                noise_b=((cb@(b*b).sum(1))-N*(mean_b*mean_b).sum(1))/(N*(N-1))
                naive=((mean_a+beta*mean_b)**2).sum(1)
                est=naive-noise_a-beta**2*noise_b
                empirical_mean=float(est.mean()); empirical_var=float(est.var(ddof=1))
                terms=exact_variance(mu,sa,sb,n,N,beta)
                mcse_mean=float(np.sqrt(terms['total']/R))
                fourth=float(np.mean((est-empirical_mean)**4))
                mcse_var=float(np.sqrt(max(0,fourth-((R-3)/(R-1))*empirical_var**2)/R))
                output.append({'case':case,'n':n,'N':N,'beta':beta,'repetitions':R,'truth':truth,
                               'mean':empirical_mean,'bias':empirical_mean-truth,'mean_mcse':mcse_mean,
                               'bias_z':(empirical_mean-truth)/mcse_mean,'empirical_variance':empirical_var,
                               'variance_mcse':mcse_var,'variance_z':(empirical_var-terms['total'])/mcse_var,
                               'variance_ratio':empirical_var/terms['total'],'negative_fraction':float(np.mean(est<0)),
                               'naive_bias':float(naive.mean()-truth),'naive_bias_theory':float(np.trace(sa)/n+beta**2*np.trace(sb)/N),
                               **{f'variance_{k}':v for k,v in terms.items()}})
    dest=ROOT/'outputs/statistical'; dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(output).to_csv(dest/'metrics.csv',index=False)
    write_json(dest/'execution.json',{'started_at':started,'completed_at':utcnow(),'specifications':specs,'settings':len(output),
                                   'max_abs_bias_z':max(abs(v['bias_z']) for v in output),
                                   'max_abs_variance_z':max(abs(v['variance_z']) for v in output)})
    print(f'Statistical experiment: {len(output)} settings x {R} repetitions',flush=True)


if __name__=='__main__':
    run()
