"""Joint NGC+SGC fit of the marked three-point function (Section 5.4).

Reads the GRAMSCI runs in ../data/3pcf: LRG_<cap>_w10_jk.3pcf (unit weights,
zeta_nnn) and LRG_<cap>_w01_jk.3pcf (luminosity weights, zeta_LLL) for the
two caps, each with its delete-one jackknife realisations in the .jk file.

The difference Delta-zeta = zeta_LLL - zeta_nnn is formed realisation by
realisation before the covariance is estimated, so the sample variance shared
by the two weightings cancels, as for Delta-xi. The covariance of each cap is
the jackknife covariance
    C = (N - 1)/N sum_k (x_k - xbar)(x_k - xbar)^T
with the Hartlap factor applied to its inverse. The caps keep separate region
sets (400 for the NGC, 211 for the SGC) and are fitted together with a
block-diagonal covariance, which is exact because the caps are disjoint on the
sky. Triangle configurations with r1 + r2 < r3 have RRR = 0 and are masked.

The fitted form is Delta-zeta = A zeta_nnn, with one amplitude A shared by the
two caps, so the fit has (number of configurations - 1) degrees of freedom.
The script also gives A for each cap separately, the chi^2 of Delta-zeta
against zero, and the linear-order expectation 3 db/b, with
b_L/b_n = B_L,t/B_n,t taken from the weighted medians of the fiducial chain.
That expectation holds only if the two weightings share a quadratic bias, so
the amplitude itself is not used as a test of the model.

Usage: python fit_3pcf.py [--out FILE]

With --out, the results are also written to FILE as an npz with the keys of
data/3pcf/zeta_caps.npz.
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import chi2 as chi2dist

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
sys.path.insert(0, HERE)
from model import hartlap  # noqa: E402
from likelihood import jackknife_cov  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=None, help='write the results to this npz')
OUT = ap.parse_args().out


def load(cap):
    """zeta_nnn, zeta_LLL and their jackknife realisations for one cap."""
    out = {}
    for tag, key in (('w10', 'nnn'), ('w01', 'LLL')):
        f = os.path.join(D, '3pcf', f'LRG_{cap}_{tag}_jk.3pcf')
        m = np.loadtxt(f, skiprows=1)
        j = np.loadtxt(f + '.jk', skiprows=2)
        out[key] = m[:, 8]
        out['jk_' + key] = j[:, 9].reshape(len(m), len(j) // len(m))
    out['nreal'] = out['jk_nnn'].shape[1]
    # a configuration is kept if it is finite in the measurement and in
    # every jackknife realisation of both weightings
    out['ok'] = np.all([np.isfinite(out[k]) if out[k].ndim == 1
                        else np.isfinite(out[k]).all(axis=1)
                        for k in ('nnn', 'LLL', 'jk_nnn', 'jk_LLL')], axis=0)
    return out


def block(c):
    """Data vector, template and inverse covariance of one cap."""
    ok, nr = c['ok'], c['nreal']
    d = (c['LLL'] - c['nnn'])[ok]
    t = c['nnn'][ok]
    jd = (c['jk_LLL'] - c['jk_nnn'])[ok]
    nb = int(ok.sum())
    Ci = hartlap(nr, nb) * np.linalg.inv(jackknife_cov(jd.T))
    return d, t, Ci, nb


def fit(blocks):
    """Amplitude A shared by the blocks, its error, chi^2 and dof."""
    num = sum(t @ Ci @ d for d, t, Ci, _ in blocks)
    den = sum(t @ Ci @ t for _, t, Ci, _ in blocks)
    A = float(num / den)
    x2 = sum(float((d - A * t) @ Ci @ (d - A * t)) for d, t, Ci, _ in blocks)
    dof = sum(nb for *_, nb in blocks) - 1
    return A, float(1 / np.sqrt(den)), x2, dof


def jkerr(x, nr):
    """Delete-one jackknife error of each row of x (bins, realisations)."""
    return np.sqrt((nr - 1) / nr * ((x - x.mean(1, keepdims=True)) ** 2).sum(1))


caps = {cap: load(cap) for cap in ('NGC', 'SGC')}
blocks = {cap: block(c) for cap, c in caps.items()}

print('Marked three-point function, Delta-zeta = A zeta_nnn')
res = {}
for cap, c in caps.items():
    ok, nr = c['ok'], c['nreal']
    canc = np.median(jkerr((c['jk_LLL'] - c['jk_nnn'])[ok], nr)
                     / np.hypot(jkerr(c['jk_nnn'][ok], nr),
                                jkerr(c['jk_LLL'][ok], nr)))
    A, sA, x2, dof = fit([blocks[cap]])
    res[cap] = (A, sA)
    print(f'  {cap}: {int(ok.sum())} of {len(ok)} configurations, {nr} jackknife '
          f'regions')
    print(f'       A = {A:.4f} +- {sA:.4f}   chi2 = {x2:.1f} / {dof}')
    print(f'       err(Delta-zeta) / hypot(err_nnn, err_LLL) = {canc:.2f} (median)')

(An, sn), (As, ss) = res['NGC'], res['SGC']
sd = np.hypot(sn, ss)
nsig = abs(An - As) / sd
print(f'  NGC - SGC: {An - As:+.4f} +- {sd:.4f} ({nsig:.2f} sigma)')

A, sA, x2, dof = fit(list(blocks.values()))
print(f'\n  joint: A = {A:.4f} +- {sA:.4f}   chi2 = {x2:.1f} / {dof}')
print(f'  error relative to the NGC alone: {sA / sn:.3f} '
      f'({100 * (1 - sA / sn):.0f} per cent smaller)')

det = sum(float(d @ Ci @ d) for d, _, Ci, _ in blocks.values())
ndet = sum(nb for *_, nb in blocks.values())
print(f'\n  Delta-zeta against zero: chi2 = {det:.1f} / {ndet}, '
      f'p = {chi2dist.sf(det, ndet):.2g}')

ch = np.load(os.path.join(D, 'chains', 'chain_mp_lumw1c_nocip_q140_fog.npz'))
nm, act = list(ch['names']), list(ch['active'])
w = ch['weights'] / ch['weights'].sum()


def median(key):
    """Weighted median of one parameter of the fiducial chain."""
    x = ch['samples'][:, act.index(nm.index(key))].astype(float)
    o = np.argsort(x)
    return float(np.interp(0.5, np.cumsum(w[o]), x[o]))


twopt = median('BLt') / median('Bnt')
pred = 3 * (twopt - 1)
rat = (1 + A) ** (1 / 3)
rat_lo, rat_hi = (1 + A - sA) ** (1 / 3), (1 + A + sA) ** (1 / 3)
print(f'\n  b_L/b_n implied by A, (1 + A)^(1/3) = {rat:.4f} '
      f'+{rat_hi - rat:.4f} -{rat - rat_lo:.4f}')
print(f'  b_L/b_n from the two-point fit, B_L,t/B_n,t = {twopt:.4f}')
print(f'  linear-order expectation 3 db/b = {pred:.4f}, '
      f'{(pred - A) / sA:.2f} sigma above A')

if OUT:
    np.savez(OUT, A=A, sigA=sA, chi2=x2, dof=dof, det=det, ndet=ndet,
             ratio=rat, sig_ratio=0.5 * (rat_hi - rat_lo), pred=pred,
             twopt=twopt, A_NGC=An, sig_NGC=sn, A_SGC=As, sig_SGC=ss,
             nsig_caps=nsig, ntri_NGC=int(caps['NGC']['ok'].sum()),
             ntri_SGC=int(caps['SGC']['ok'].sum()),
             njk_NGC=caps['NGC']['nreal'], njk_SGC=caps['SGC']['nreal'])
    print(f'\nwrote {OUT}')
