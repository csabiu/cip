"""Data vector, covariance, priors and likelihood of the LRG multipole fits.

The data vector is [xi_0,n, xi_2,n, xi_0,L, xi_2,L], with xi_0 over
40-200 Mpc/h and xi_2 over 40-140 Mpc/h: 104 points. The covariance is the
jackknife covariance of the 400 realisations,
    C = (N - 1)/N sum_k (x_k - xbar)(x_k - xbar)^T,
and its inverse is multiplied by the Hartlap factor (N - p - 2)/(N - 1).
With mockcal=True the inverse is further divided by r^2, where r is the ratio
of the scatter of B_L,Delta over the EZmocks to its jackknife error
(Appendix B), as computed by calibrate_covariance.py.

The model is model.model_fog. Besides the fiducial parametrisation, the
likelihood supports the variants whose results the paper quotes:
  fix_bld      B_L,Delta = 0 (null model of the Bayes factor)
  ratio=R      B_L,Delta = R B_n,Delta, with R = 1.017 from the selection
  sumdiff      sample S = (B_L,Delta + B_n,Delta)/2 and D = B_L,Delta - B_n,Delta
  cip_positive B_CIP free with 0 <= B_CIP <= 0.3
  product      sample v = B_CIP (B_n,Delta^2 + B_L,Delta^2) in [0, vmax]
"""
import os

import numpy as np

from model import TemplatesFoG, hartlap, model_fog

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
SMIN, SMAX0, SMAX2 = 40.0, 200.0, 140.0
VECTORS = ('lrg_multipoles.npz', 'lrg_injected.npz')

# uniform priors of the fiducial fit (Table 1 of the paper); B_CIP = 0
PRIORS = {
    'Bnt': (0.0, 5.0), 'BLt': (0.0, 5.0),
    'BnD': (-10.0, 10.0), 'BLD': (-10.0, 10.0),
    'Bcip': (0.0, 0.0),
    'Bsys_n': (-0.0015, 0.0015), 'Bsys_L': (-0.01, 0.01),
    'Bsys2_n': (-0.0015, 0.0015), 'Bsys2_L': (-0.01, 0.01),
    'Amc': (0.0, 6.0), 'kstar': (0.0, 10.0), 'sigv': (0.0, 12.0),
}
NAMES = list(PRIORS)
IBN, IBL, ICIP = NAMES.index('BnD'), NAMES.index('BLD'), NAMES.index('Bcip')


def field(name):
    """multipoles (2, 32) and jackknife realisations (400, 2, 32) of one field,
    from lrg_multipoles.npz or, for injected fields, lrg_injected.npz"""
    for f in VECTORS:
        d = np.load(os.path.join(DATA, f))
        if f'xi_{name}' in d.files and f'jk_{name}' in d.files:
            return d['s'], d[f'xi_{name}'], d[f'jk_{name}']
    raise KeyError(f'no field {name} in {VECTORS}')


def data_vector(mark='lumw1c', num='num'):
    """data vector, jackknife realisations and the s values of each block"""
    s, xn, jn = field(num)
    _, xl, jl = field(mark)
    c0 = (s >= SMIN) & (s <= SMAX0)
    c2 = (s >= SMIN) & (s <= SMAX2)

    def vec(x):
        return np.concatenate([x[0][c0], x[1][c2]])

    y = np.concatenate([vec(xn), vec(xl)])
    jk = np.array([np.concatenate([vec(a), vec(b)]) for a, b in zip(jn, jl)])
    return y, jk, s[c0], s[c2]


def jackknife_cov(jk):
    n = len(jk)
    r = jk - jk.mean(0)
    return (n - 1) / n * r.T @ r


def calibration_factor(path=None):
    """r of Appendix B: the mean over the 25 mocks of the ratio for B_L,Delta"""
    c = np.load(path or os.path.join(DATA, 'ezmock', 'jk_covtest.npz'))
    return float(c['par_hart_ratio'][:, list(c['names']).index('BLD')].mean())


class Likelihood:
    """Gaussian likelihood of the fiducial model (fingers-of-god, xi_2 to 140)."""

    def __init__(self, mark='lumw1c', num='num', fix_bld=False, mockcal=False,
                 r=None, ratio=None, sumdiff=False, cip_positive=False,
                 product=False, bldrange=10.0, sumrange=30.0, difrange=10.0,
                 vmax=0.05):
        if sum([fix_bld, ratio is not None, sumdiff, product]) > 1:
            raise ValueError('fix_bld, ratio, sumdiff and product exclude each other')
        self.y, jk, self.s0, self.s2 = data_vector(mark, num)
        self.nreal = len(jk)
        self.cinv = hartlap(self.nreal, self.y.size) * np.linalg.inv(jackknife_cov(jk))
        self.r = 1.0
        if mockcal:
            self.r = calibration_factor() if r is None else r
            self.cinv = self.cinv / self.r ** 2
        tf = os.path.join(DATA, 'templates', 'templates_fog.npz')
        self.tpl = TemplatesFoG(tf)
        self.xi_cip = np.load(tf)['xi_cip']

        pr = dict(PRIORS, BnD=(-bldrange, bldrange), BLD=(-bldrange, bldrange))
        if fix_bld or ratio is not None:
            pr['BLD'] = (0.0, 0.0)          # set from B_n,Delta for ratio
        if sumdiff:                         # the two slots carry S and D
            pr['BnD'], pr['BLD'] = (-sumrange, sumrange), (-difrange, difrange)
        if cip_positive:
            pr['Bcip'] = (0.0, 0.3)
        if product:                         # the B_CIP slot carries v
            pr['Bcip'] = (0.0, vmax)
        self.ratio, self.sumdiff, self.product = ratio, sumdiff, product
        self.lo = np.array([pr[k][0] for k in NAMES])
        self.hi = np.array([pr[k][1] for k in NAMES])
        self.active = np.flatnonzero(self.hi > self.lo)
        self.ndim = len(self.active)

    def theta(self, u):
        """model parameters from the sampled (active) parameters"""
        th = self.lo.copy()
        th[self.active] = u
        if self.ratio is not None:
            th[IBL] = self.ratio * th[IBN]
        if self.sumdiff:
            S, D = th[IBN], th[IBL]
            th[IBN], th[IBL] = S - 0.5 * D, S + 0.5 * D
        if self.product:
            q = th[IBN] ** 2 + th[IBL] ** 2
            th[ICIP] = th[ICIP] / q if q > 1e-12 else 0.0
        return th

    def prior_transform(self, q):
        return self.lo[self.active] + q * (self.hi[self.active] - self.lo[self.active])

    def chi2(self, u):
        r = self.y - model_fog(self.theta(u), self.tpl, self.s0, self.s2, self.xi_cip)
        return float(r @ self.cinv @ r)

    def __call__(self, u):
        if self.product and self.theta(u)[ICIP] > 0.3:
            return -1e100       # keep B_CIP inside its prior range
        return -0.5 * self.chi2(u)
