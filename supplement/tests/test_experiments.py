import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def load(name):
    assert (ROOT/f'{name}.py').exists(), f'{name} implementation missing'
    return __import__(name)


def test_candidate_snapshots_match_original_training():
    t = load('training')
    from synrcp.v11_neural import fit_threshold_network
    rng = np.random.default_rng(31)
    x,xp = rng.normal(size=(20,3)),rng.normal(size=(40,3))
    s,sd,sp = rng.random(20),rng.random(20),rng.random(40)
    fits = t.fit_candidates(x,s,xp,sp,sd,powers=[0.,1.],checkpoints=[2,4],seed=4,tau=.9,hidden=8,learning_rate=.002,bound=2.)
    for j,power in enumerate([0.,1.]):
        for k,epoch in enumerate([2,4]):
            original = fit_threshold_network(x,s,tau=.9,seed=4,objective='supervised' if power==0 else 'signed',power=power,x_pool=xp if power else None,score_pool=sp if power else None,score_paired=sd if power else None,hidden=8,epochs=epoch,learning_rate=.002,output_bound=2.)
            np.testing.assert_array_equal(fits[j*2+k]['predict'](x),original.predict(x))


def test_tabular_roles_disjoint_and_original_prefixes_preserved():
    r = load('run_tabular')
    from synrcp.v11 import _allocate_v11_roles,_prefix
    original = _allocate_v11_roles(17379,1000)
    roles = r.allocate_roles(17379,0)
    np.testing.assert_array_equal(roles['pool_train'],_prefix(original['synthetic_pool'],17379,.3))
    np.testing.assert_array_equal(np.r_[roles['learning'],roles['calibration']],_prefix(original['conformal'],17379,.01))
    values = np.concatenate(list(roles.values()))
    assert len(np.unique(values)) == len(values)
    assert len(roles['development']) > 0 and len(roles['pool_validation']) > 0


def test_exact_quadratic_variance_by_exhaustive_enumeration():
    st = load('run_statistics')
    a = np.array([[-.4,.1],[.6,-.2]])
    b = np.array([[.2,.4],[-.2,-.1]])
    pa,pb = np.array([.3,.7]),np.array([.6,.4])
    beta=.7
    mu=pa@a+beta*(pb@b)
    ca=(a-pa@a).T@((a-pa@a)*pa[:,None])
    cb=(b-pb@b).T@((b-pb@b)*pb[:,None])
    vals,probs=[],[]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    vals.append(a[i]@a[j]+beta**2*b[k]@b[l]+2*beta*((a[i]+a[j])/2)@((b[k]+b[l])/2))
                    probs.append(pa[i]*pa[j]*pb[k]*pb[l])
    vals,probs=np.array(vals),np.array(probs)
    assert probs@vals == pytest.approx(mu@mu)
    assert probs@((vals-mu@mu)**2) == pytest.approx(st.exact_variance(mu,ca,cb,2,2,beta)['total'])
