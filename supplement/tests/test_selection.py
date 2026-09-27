import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "postconformal_selection.py"
    assert path.exists(), "selection implementation has not been added"
    spec = importlib.util.spec_from_file_location("selection", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exact_order_statistic_and_infinity():
    m = module()
    assert m.conformal_offset(np.arange(9.), .9) == 8
    assert np.isinf(m.conformal_offset(np.arange(8.), .9))
    with pytest.raises(ValueError):
        m.conformal_offset([], .9)


def test_quadratic_is_cross_pair_ustatistic_and_can_be_negative():
    m = module()
    z = np.array([-.9, .1, .1, .1])
    phi = np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
    zd = np.array([.1,-.9,.1,.1])
    bphi = np.array([[1.,0.],[0.,1.],[1.,0.]])
    bz = np.array([.1,.1,-.9])
    for beta in [0., .5, 1.]:
        a = phi*(z-beta*zd)[:,None]
        b = bphi*bz[:,None]
        expected = sum(a[i]@a[j] for i in range(4) for j in range(4) if i!=j)/12
        expected += beta**2*sum(b[i]@b[j] for i in range(3) for j in range(3) if i!=j)/6
        expected += 2*beta*a.mean(0)@b.mean(0)
        actual = m.quadratic_moment_estimate(z,phi,beta=beta,z_synthetic=zd,pool_phi=bphi,pool_z_synthetic=bz)
        assert actual == pytest.approx(expected)
    assert m.quadratic_moment_estimate(np.array([-1.,1.]), np.ones((2,1))) == -1


def test_beta_zero_ignores_invalid_synthetic_values():
    m = module()
    assert np.isfinite(m.quadratic_moment_estimate(np.array([.1,.1]),np.ones((2,1)),z_synthetic=np.array([np.nan])))
    with pytest.raises(ValueError):
        m.quadratic_moment_estimate(np.ones(2),np.ones((2,1)),beta=1)


def test_shift_invariance_and_no_final_inputs():
    m = module()
    rng = np.random.default_rng(881)
    n, N, d, M = 100, 500, 3, 4
    x, xp = rng.normal(size=(n,d)), rng.normal(size=(N,d))
    w = rng.normal(size=(d,5))
    phi, pp = m.bounded_fourier_features(x,w), m.bounded_fourier_features(xp,w)
    np.testing.assert_allclose(np.sum(phi**2,axis=1),1)
    q, qp = rng.normal(size=(M,n)), rng.normal(size=(M,N))
    s, sd, sp = rng.normal(size=n), rng.normal(size=n), rng.normal(size=N)
    a = m.selection_scores(q,s,phi,sdag_dev=sd,q_pool=qp,sdag_pool=sp,phi_pool=pp,seed=31)
    shifts = np.array([-2.,-.5,.5,2.])
    b = m.selection_scores(q+shifts[:,None],s,phi,sdag_dev=sd,q_pool=qp+shifts[:,None],sdag_pool=sp,phi_pool=pp,seed=31)
    for key in ['calibrated_pinball','coverage_beta0','coverage_beta1']:
        np.testing.assert_allclose(a['scores'][key],b['scores'][key],atol=1e-12)
    assert not np.allclose(a['scores']['raw_pinball'],b['scores']['raw_pinball'])
    assert len(set(a['fold0']) & set(a['fold1'])) == 0
    assert sorted(a['fold0']+a['fold1']) == list(range(n))


def test_reject_bad_shapes_and_small_development():
    m = module()
    with pytest.raises(ValueError):
        m.selection_scores(np.zeros((2,3)),np.zeros(3),np.ones((3,1)))
    with pytest.raises(ValueError):
        m.quadratic_moment_estimate(np.zeros(3),np.ones((2,1)))
