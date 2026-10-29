"""Calibration of the jackknife covariance against the EZmocks (Appendix B).

The jackknife of the fiducial fit (400 regions) was repeated on 25 of the
1000 DR1 EZmocks. Each of the 25 jackknife covariances C_i is used in turn as
the covariance of a fit to the mocks, and the scatter of the fitted
parameters over the mocks is compared with their quoted errors.

The fit is the fiducial model of likelihood.py, linearised about the
weighted posterior median theta0 of the fiducial chain, with the Jacobian J
taken by central differences. For the weight W = h C_i^-1, with h the Hartlap
factor, the generalised least-squares estimate has quoted covariance
Q = (J^T W J)^-1 and true covariance S = Q J^T W C W J Q, where C is the
covariance of the other 999 mocks. The ratio sqrt(S/Q) of realised to quoted
error, averaged over the 25 jackknifes, is r. For B_L,Delta it is the factor
by which the paper inflates the Hartlap-corrected intervals of the LRG
multipole fits; r/sqrt(m1), with m1 the Percival et al. (2014) factor, is the
part that the Percival correction does not account for.

Also computed:
  noise h   tr(C_i^-1 Cbar)/p, with Cbar the mean of the other 24 jackknifes:
            the inverse bias of a single jackknife covariance, which for a
            Wishart matrix with nu degrees of freedom is nu/(nu - p - 1);
  split     the same error ratio, with the Hartlap and Percival factors, for
            a covariance from a random half of the mocks applied to the other
            half, and <chi^2>/p of that half. For a Wishart covariance both
            are 1, so they test the method itself.

Usage: python calibrate_covariance.py [--out FILE] [--nsplit 50]

Prints the error ratios of every parameter and of B_L,Delta - B_n,Delta, and
writes FILE (default ../runs/covariance_calibration.npz) with the keys of
data/ezmock/jk_covtest.npz read by likelihood.calibration_factor() and
compute_results.py: names, par_hart_ratio (mock, quantity) and percival_m1,
together with hartlap, theta0, jk_ids, noise_h, cal_ratio and cal_chi2.
"""
import argparse
import os
import sys

import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path.insert(0, HERE)
from model import TemplatesFoG, hartlap, model_fog, percival  # noqa: E402
from likelihood import NAMES, SMAX0, SMAX2, SMIN, jackknife_cov  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(HERE, '..', 'runs', 'covariance_calibration.npz'),
                help='output file')
ap.add_argument('--nsplit', type=int, default=50, help='number of random half splits')
args = ap.parse_args()

# data vectors [xi_0,n, xi_2,n, xi_0,L, xi_2,L] of the mocks
mm = np.load(os.path.join(DATA, 'ezmock', 'mock_multipoles.npz'))
mj = np.load(os.path.join(DATA, 'ezmock', 'mock_jackknife.npz'))
s = mm['s']
c0, c2 = (s >= SMIN) & (s <= SMAX0), (s >= SMIN) & (s <= SMAX2)


def vec(xn, xl):
    """multipoles (..., 2, nbin) of the number- and luminosity-weighted fields
    -> (..., 104) data vector"""
    xn, xl = xn.astype(np.float64), xl.astype(np.float64)
    return np.concatenate([xn[..., 0, c0], xn[..., 1, c2],
                           xl[..., 0, c0], xl[..., 1, c2]], axis=-1)


ids = mm['ids']
Y = vec(mm['x'][:, 0], mm['x'][:, 1])                       # (1000, 104)
jk_ids = mj['ids']
Cj = np.array([jackknife_cov(vec(a, b)) for a, b in zip(mj['jk_n'], mj['jk_L'])])
N, P = Y.shape
NJK = mj['jk_n'].shape[1]

# fiducial model, linearised about the posterior median
tf = os.path.join(DATA, 'templates', 'templates_fog.npz')
tpl, xi_cip = TemplatesFoG(tf), np.load(tf)['xi_cip']
ch = np.load(os.path.join(DATA, 'chains', 'chain_mp_lumw1c_nocip_q140_fog.npz'))
act = list(ch['active'])
w = ch['weights'].astype(np.float64)
w = w / w.sum()


def wquant(x, q):
    o = np.argsort(x)
    return np.interp(q, np.cumsum(w[o]) - 0.5 * w[o], x[o])


theta0, step = np.zeros(len(NAMES)), np.zeros(len(NAMES))  # B_CIP stays 0
for k, a in enumerate(act):
    lo, med, hi = wquant(ch['samples'][:, k].astype(np.float64), [0.16, 0.5, 0.84])
    theta0[a], step[a] = med, 0.1 * (hi - lo) / 2


def model(th):
    return model_fog(th, tpl, s[c0], s[c2], xi_cip)


J = np.empty((P, len(act)))
for k, a in enumerate(act):
    d = np.zeros(len(NAMES))
    d[a] = step[a]
    J[:, k] = (model(theta0 + d) - model(theta0 - d)) / (2 * step[a])

# quantities: every free parameter, and B_L,Delta - B_n,Delta
NP = len(act)
pn = [NAMES[a] for a in act]
G = np.vstack([np.eye(NP), np.eye(NP)[pn.index('BLD')] - np.eye(NP)[pn.index('BnD')]])
names = pn + ['BLD-BnD']


def error_ratio(K, C, c):
    """realised/quoted error of each quantity, for the weight c K^-1 and
    data of covariance C"""
    W = c * np.linalg.inv(K)
    Q = np.linalg.inv(J.T @ W @ J)
    S = Q @ J.T @ W @ C @ W @ J @ Q
    return np.sqrt(np.einsum('ij,jk,ik->i', G, S, G) / np.einsum('ij,jk,ik->i', G, Q, G))


def trace_ratio(K, C):
    return np.trace(np.linalg.solve(K, C)) / P


# per-mock jackknifes
hart, m1 = hartlap(NJK, P), percival(NJK, P, NP)
ratio = np.array([error_ratio(Cj[i], np.cov(Y[ids != j], rowvar=False), hart)
                  for i, j in enumerate(jk_ids)])        # (25, quantities)
NM = len(Cj)
Cbar = Cj.mean(0)
noise_h = np.array([trace_ratio(Cj[i], (NM * Cbar - Cj[i]) / (NM - 1)) for i in range(NM)])

# half splits of the mocks
rng = np.random.default_rng(1)
NA = N // 2
hA, m1A = hartlap(NA, P), percival(NA, P, NP)
cal_ratio, cal_chi2 = [], []
for _ in range(args.nsplit):
    o = rng.permutation(N)
    CA, CB = np.cov(Y[o[:NA]], rowvar=False), np.cov(Y[o[NA:]], rowvar=False)
    cal_ratio.append(error_ratio(CA, CB, hA) / np.sqrt(m1A))
    cal_chi2.append(hA * trace_ratio(CA, CB))
cal_ratio = np.array(cal_ratio).reshape(-1, len(names))
cal_chi2 = np.array(cal_chi2)


# summary
def ms(a):
    return f'{a.mean():.3f} +- {a.std(ddof=1) / np.sqrt(len(a)):.3f}'


print(f'{N} mocks, {NM} per-mock jackknifes of {NJK} regions, '
      f'{P} data points, {NP} parameters')
print(f'Hartlap factor {hart:.4f}, Percival m1 {m1:.4f} (sqrt {np.sqrt(m1):.4f})')
print()
print('realised/quoted error, mean +- error over the per-mock jackknifes')
print(f'  {"":8s} {"raw":>14s} {"Hartlap":>14s} {"H+Percival":>14s} '
      f'{"cov. raw":>9s} {"cov. H+P":>9s} {"split":>6s}')
raw, hp = ratio / np.sqrt(hart), ratio / np.sqrt(m1)
for q, nm in enumerate(names):
    cr = erf(1 / (np.sqrt(2) * raw[:, q])).mean()
    cov = erf(1 / (np.sqrt(2) * hp[:, q])).mean()
    sp = f'{cal_ratio[:, q].mean():6.3f}' if args.nsplit else ''
    print(f'  {nm:8s} {ms(raw[:, q]):>14s} {ms(ratio[:, q]):>14s} {ms(hp[:, q]):>14s} '
          f'{cr:9.3f} {cov:9.3f} {sp}')
print('  (cov.: mean 1-sigma coverage; split: covariance from half of the mocks)')
print()
r = ratio[:, names.index('BLD')].mean()
rd = ratio[:, names.index('BLD-BnD')].mean()
nu = NJK - 1
print(f'r = {r:.4f} for B_L,Delta ({rd:.4f} for B_L,Delta - B_n,Delta)')
print(f'r/sqrt(m1) = {r / np.sqrt(m1):.4f} for B_L,Delta ({rd / np.sqrt(m1):.4f})')
print(f'noise h = {ms(noise_h)} (Wishart with nu = {nu}: {nu / (nu - P - 1):.3f}), '
      f'effective nu = {noise_h.mean() * (P + 1) / (noise_h.mean() - 1):.0f}')
if args.nsplit:
    print(f'split <chi^2>/p = {ms(cal_chi2)} over {args.nsplit} splits')

out = args.out if args.out.endswith('.npz') else args.out + '.npz'
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
np.savez(out, names=np.array(names), par_hart_ratio=ratio, percival_m1=m1,
         hartlap=hart, theta0=theta0, jk_ids=jk_ids, noise_h=noise_h,
         cal_ratio=cal_ratio, cal_chi2=cal_chi2)
print(f'wrote {out}')
