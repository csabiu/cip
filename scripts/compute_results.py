"""Compute Table 2 and the main constraints of the paper from ../data.

Errors on the LRG multipole fits include the calibration of the jackknife
covariance against the EZmocks (Appendix B of the paper): the Percival et al.
(2014) factor times the residual measured on the mocks. The other fits carry
the Percival factor only, and the _mockcal chains need no correction.

Usage: python compute_results.py [--chains DIR] [--calibration FILE]

With --chains, chains found in DIR (for example those written by fit.py) are
used in place of the shipped ones of the same name; with --calibration, the
factor r comes from a file written by calibrate_covariance.py.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
sys.path.insert(0, HERE)
from model import percival  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--chains', default=None, metavar='DIR',
                help='directory of chains that replace the shipped ones of the same name')
ap.add_argument('--calibration', default=None,
                help='covariance calibration written by calibrate_covariance.py, '
                     'in place of data/ezmock/jk_covtest.npz')
ARGS = ap.parse_args()
CHAINS = ARGS.chains
if CHAINS and not os.path.isdir(CHAINS):
    sys.exit(f'compute_results.py: error: --chains: no directory {CHAINS}')
TAKEN = set()

jc = np.load(ARGS.calibration or os.path.join(D, 'ezmock', 'jk_covtest.npz'))
RESID = (jc['par_hart_ratio'][:, list(jc['names']).index('BLD')].mean()
         / np.sqrt(float(jc['percival_m1'])))


def load(tag):
    if CHAINS and os.path.exists(os.path.join(CHAINS, f'chain_{tag}.npz')):
        TAKEN.add(tag)
        return np.load(os.path.join(CHAINS, f'chain_{tag}.npz'), allow_pickle=True)
    return np.load(os.path.join(D, 'chains', f'chain_{tag}.npz'), allow_pickle=True)


def col(ch, key):
    nm, act = list(ch['names']), list(ch['active'])
    if 'derived' in ch.files and key in ('BLD', 'BnD', 'Bcip'):
        return ch['derived'][:, nm.index(key)].astype(float)
    if nm.index(key) not in act:
        return None
    return ch['samples'][:, act.index(nm.index(key))].astype(float)


def factor(tag, ch):
    """error inflation for a chain: Percival, times the mock residual for
    the 400-region LRG multipole jackknife"""
    if '_mockcal' in tag:
        return 1.0
    bgs = tag.startswith('BGS')
    mp = tag.split('_')[1 if bgs else 0] == 'mp'
    nreg = 128 if bgs else (400 if mp else 256)
    if mp:
        nbin = 104 if '_q140' in tag else 128
    else:
        nbin = 44 if bgs else 64
    f = np.sqrt(percival(nreg, nbin, int(ch['ndim'])))
    return f * RESID if nreg == 400 else f


def interval(tag, x=None, key='BLD', ps=(0.16, 0.5, 0.84)):
    ch = load(tag)
    x = col(ch, key) if x is None else x
    w = ch['weights'] / ch['weights'].sum()
    o = np.argsort(x)
    q = np.interp(ps, np.cumsum(w[o]), x[o])
    return q[1], q[1] + (q - q[1]) * factor(tag, ch)


def fmt(tag, x=None):
    m, q = interval(tag, x)
    return f'{m:+.2f} +{q[2] - m:.2f} -{m - q[0]:.2f}'


ROWS = [
    ('mp_lumw1c_nocip_q140_fog', 'xi0+xi2', 'LRG', 'W1 (corr.)', 'B_CIP = 0'),
    ('mono_lumw1c_nocip', 'xi0', 'LRG', 'W1 (corr.)', 'B_CIP = 0'),
    ('mp_lumzc_nocip_q140_fog', 'xi0+xi2', 'LRG', 'z (corr.)', 'B_CIP = 0'),
    ('mono_lumzc_nocip', 'xi0', 'LRG', 'z (corr.)', 'B_CIP = 0'),
    ('mp_mstarfs_nocip_q140_fog', 'xi0+xi2', 'LRG', 'M*', 'B_CIP = 0'),
    ('mono_mstarfs_nocip', 'xi0', 'LRG', 'M*', 'B_CIP = 0'),
    ('mp_lumw1_nocip_q140_fog', 'xi0+xi2', 'LRG', 'W1 (raw)', 'B_CIP = 0'),
    ('mp_lumz_nocip_q140_fog', 'xi0+xi2', 'LRG', 'z (raw)', 'B_CIP = 0'),
    ('BGS_mono_lumr_nocip', 'xi0', 'BGS', 'r', 'B_CIP = 0'),
    ('BGS_mono_lumw1_nocip', 'xi0', 'BGS', 'W1', 'B_CIP = 0'),
    ('mp_lumw1c_cippos_q140_fog', 'xi0+xi2', 'LRG', 'W1 (corr.)', 'B_CIP >= 0'),
    ('mp_lumzc_cippos_q140_fog', 'xi0+xi2', 'LRG', 'z (corr.)', 'B_CIP >= 0'),
    ('mp_mstarfs_cippos_q140_fog', 'xi0+xi2', 'LRG', 'M*', 'B_CIP >= 0'),
    ('BGSF_mono_lumr_nocip', 'xi0', 'BGSF', 'r', 'B_CIP = 0'),
    ('BGSF_mono_lumw1_nocip', 'xi0', 'BGSF', 'W1', 'B_CIP = 0'),
]

print('Table 2')
print(f'{"statistic":9s} {"sample":6s} {"mark":11s} {"model":11s} '
      f'{"B_L,Delta":>22s} {"P(>0)":>6s} {"B_L,D^2 B_CIP":>14s} {"ln Z":>7s}')
for tag, stat, samp, mark, model in ROWS:
    ch = load(tag)
    bl, bc = col(ch, 'BLD'), col(ch, 'Bcip')
    w = ch['weights'] / ch['weights'].sum()
    if bc is None:
        b, p, prod = fmt(tag), f'{w[bl > 0].sum():.2f}', '--'
    else:
        m, q = interval(tag, bl ** 2 * bc, ps=(0.5, 0.5, 0.95))
        b, p, prod = '--', '--', f'< {q[2]:.4f}'
    print(f'{stat:9s} {samp:6s} {mark:11s} {model:11s} {b:>22s} {p:>6s} {prod:>14s} '
          f'{float(ch["logz"]):7.1f}')

FID = 'mp_lumw1c_nocip_q140_fog'
ch = load(FID)
print('\nBaseline (Section 5.2)')
print(f'  B_L,Delta              {fmt(FID)}')
m, q = interval(FID, col(ch, 'BLD') - col(ch, 'BnD'))
print(f'  B_L,Delta - B_n,Delta  {m:+.2f} +- {0.5 * (q[2] - q[0]):.2f}')
print(f'  ratio imposed          {fmt("mp_lumw1c_nocip_ratio_q140_fog")}')
sd = load('mp_lumw1c_nocip_sumdiff_q140_fog')
for lab, x in (('S = (B_L + B_n)/2', 0.5 * (col(sd, 'BLD') + col(sd, 'BnD'))),
               ('D = B_L - B_n', col(sd, 'BLD') - col(sd, 'BnD'))):
    m, q = interval('mp_lumw1c_nocip_sumdiff_q140_fog', x)
    print(f'  {lab:22s} {m:+.2f} +- {0.5 * (q[2] - q[0]):.2f}')


def lnb(a, b):
    a, b = load(a), load(b)
    return (f"{float(a['logz'] - b['logz']):+.2f} +- "
            f"{float(np.hypot(a['logzerr'], b['logzerr'])):.2f}")


print('\nBayes factors (mock-calibrated likelihood)')
print(f'  B_L,Delta free vs 0    ln B = '
      f'{lnb("mp_lumw1c_nocip_mockcal_q140_fog", "mp_lumw1c_nobld_nocip_mockcal_q140_fog")}')
print(f'  FoG vs linear Kaiser   ln B = '
      f'{lnb("mp_lumw1c_nocip_mockcal_q140_fog", "mp_lumw1c_nocip_mockcal_q140")}')

pc = load('mp_lumw1c_prod_q140_fog')
m, q = interval('mp_lumw1c_prod_q140_fog', col(pc, 'BLD') ** 2 * col(pc, 'Bcip'),
                ps=(0.5, 0.5, 0.95))
print(f'\nCIP (Section 5.3)\n  B_L,Delta^2 B_CIP < {q[2]:.3f} (95 per cent)')

z = np.load(os.path.join(D, '3pcf', 'zeta_caps.npz'))
print(f'\nMarked 3PCF (Section 5.4)\n  A = {float(z["A"]):.3f} +- {float(z["sigA"]):.3f}')
if CHAINS:
    print(f'\nChains taken from {CHAINS}: {", ".join(sorted(TAKEN)) or "none"}')
