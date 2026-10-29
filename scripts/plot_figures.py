"""Make the figures of the paper from the files in ../data.

Usage:  python plot_figures.py [figure ...]      (default: all)
        python plot_figures.py corner [tag ...]
Output: ../figures/

'corner' draws the posterior of the baseline fit to corner_baseline.png.
Followed by chain tags (file names in ../data/chains without 'chain_' and
'.npz'), it draws those fits instead, each to corner_<tag>.png.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
F = os.path.join(HERE, '..', 'figures')
os.makedirs(F, exist_ok=True)
sys.path.insert(0, HERE)
from model import TemplatesFoG, model_fog, PARAMS_FOG, hartlap  # noqa: E402

LB = {'lumw1c': 'W1 (corr.)', 'lumzc': r'$z$-band (corr.)', 'lumw1': 'W1 (raw)',
      'lumz': r'$z$-band (raw)', 'mstarfs': r'$M_\ast$', 'lumr': r'$r$-band'}
FIDUCIAL = 'mp_lumw1c_nocip_q140_fog'
# axis labels of the chain columns; bb<k>_<f> are the broadband terms a_k,f
TEX = {'Bnt': r'$B_{{\rm n},t}$', 'BLt': r'$B_{{\rm L},t}$', 'BnD': r'$B_{\rm n,\Delta}$',
       'BLD': r'$B_{\rm L,\Delta}$', 'Bcip': r'$B_{\rm CIP}$',
       'Bsys_n': r'$B_{\rm sys,n}$', 'Bsys_L': r'$B_{\rm sys,L}$',
       'Bsys2_n': r'$B_{\rm sys2,n}$', 'Bsys2_L': r'$B_{\rm sys2,L}$',
       'Amc': r'$A_{\rm MC}$', 'Amc_L': r'$A_{\rm MC,L}$', 'kstar': r'$k_\ast$',
       'kstar_L': r'$k_{\ast,\rm L}$', 'sigv': r'$\sigma_{\rm v}$', 'Sperp': r'$\Sigma_\perp$',
       'Sperp_L': r'$\Sigma_{\perp,\rm L}$'}
TEX.update({f'bb{k}_{f}': rf'$a_{{{k},\rm {f}}}$' for k in range(3) for f in 'nL'})


def chain(tag):
    return np.load(os.path.join(D, 'chains', f'chain_{tag}.npz'), allow_pickle=True)


def median(ch, key):
    """weighted posterior median; parameters fixed in a variant are zero"""
    nm, act = list(ch['names']), list(ch['active'])
    if nm.index(key) not in act:
        return 0.0
    v = ch['samples'][:, act.index(nm.index(key))]
    o = np.argsort(v)
    return float(np.interp(0.5, np.cumsum(ch['weights'][o] / ch['weights'].sum()), v[o]))


def jk_err(jk):
    """jackknife standard deviation over the leading (realisation) axis"""
    n = len(jk)
    return np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0))


def samples():
    return np.load(os.path.join(D, 'samples.npz'))


def fig_data():
    d = samples()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for pref, c in (('LRG', 'C0'), ('BGS', 'C1'), ('BGSF', 'C2')):
        if f'{pref}_z' not in d:
            continue
        ax[0].semilogy(d[f'{pref}_z'], d[f'{pref}_n'], '-', color=c, lw=1.8,
                       label=f'{pref}  ({int(d[f"{pref}_N"])/1e3:.0f}k)')
    ax[0].set_xlabel('$z$')
    ax[0].set_ylabel(r'$\bar n(z)\ [h^{3}\,{\rm Mpc}^{-3}]$')
    ax[0].legend(fontsize=8.5)
    ax[0].text(0.03, 0.05, '(a)', transform=ax[0].transAxes, fontsize=11)

    if 'Mr' in d:
        ax[1].plot(d['z'], d['Mr'], ',', color='0.6', rasterized=True)
        ax[1].axhline(-20.5, color='C2', lw=1.5, ls='--')
        ax[1].axvline(0.195, color='C2', lw=1.5, ls='--', label='BGSF cut')
        ax[1].set_xlabel('$z$'); ax[1].set_ylabel('$M_r$')
        ax[1].set_ylim(-16, -24); ax[1].set_xlim(0, 0.45)
        ax[1].legend(fontsize=8.5, loc='lower right')
    ax[1].text(0.03, 0.05, '(b)', transform=ax[1].transAxes, fontsize=11)

    e = d['mark_bins']
    xc = 0.5 * (e[1:] + e[:-1])
    for key, lab, c, ls in (('LRG_mark_w1', 'LRG W1', 'C0', '-'),
                            ('LRG_mark_z', r'LRG $z$-band', 'C0', '--'),
                            ('BGSF_mark_r', r'BGSF $r$-band', 'C2', '-'),
                            ('LRG_mstarfs', r'LRG $M_\ast$', 'C3', ':')):
        if f'{key}_pdf' not in d:
            continue
        ax[2].plot(xc, d[f'{key}_pdf'], ls, color=c, lw=1.7,
                   label=rf'{lab}  $\sigma_\ell={float(d[f"{key}_sigma"]):.2f}$')
    ax[2].axvline(1.0, color='gray', lw=0.7)
    ax[2].set_xlabel(r'mark $\ell$'); ax[2].set_ylabel(r'$p(\ell)$')
    ax[2].legend(fontsize=8.5)
    ax[2].text(0.03, 0.05, '(c)', transform=ax[2].transAxes, fontsize=11)
    fig.tight_layout(); save(fig, 'data_overview')


def fig_dxi():
    d = np.load(os.path.join(D, 'lrg_monopole.npz'))
    s = d['s']
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    ax[0].plot(s, s**2 * d['xi_num'], 'k-o', ms=4, lw=1.2, label=r'$\xi_{\rm n}$')
    ax[0].plot(s + 1, s**2 * d['xi_lumzc'], 'C0--s', ms=4, lw=1.2,
               label=r'$\xi_{\rm L}$ ($z$-band)')
    ax[0].set_xlabel(r'$s\ [h^{-1}{\rm Mpc}]$')
    ax[0].set_ylabel(r'$s^2\,\xi(s)\ [h^{-2}{\rm Mpc}^2]$')
    ax[0].legend()
    ax[1].axhline(0, color='gray', lw=0.8)
    for mk, c, m in (('lumzc', 'C0', 's'), ('lumw1c', 'C1', '^')):
        ax[1].errorbar(s, s**2 * d[f'dxi_{mk}'], s**2 * d[f'dxi_{mk}_err'],
                       fmt=c + m + '-', ms=4, lw=1.2, capsize=2, label=LB[mk])
    ax[1].set_xlabel(r'$s\ [h^{-1}{\rm Mpc}]$')
    ax[1].set_ylabel(r'$s^2\,[\xi_{\rm L}-\xi_{\rm n}]$')
    ax[1].legend()
    fig.tight_layout(); save(fig, 'delta_xi_jk')


def fig_detrend():
    d = np.load(os.path.join(D, 'lrg_monopole.npz'))
    s = d['s']
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    for a, (raw, cor, lab) in zip(ax, [('lumz', 'lumzc', r'$z$-band'),
                                       ('lumw1', 'lumw1c', 'W1')]):
        a.axhline(0, color='gray', lw=0.8)
        a.errorbar(s, s**2 * d[f'dxi_{raw}'], s**2 * d[f'dxi_{raw}_err'],
                   fmt='o-', ms=3, capsize=2, color='C3', label=f'{lab} raw')
        a.errorbar(s + 1, s**2 * d[f'dxi_{cor}'], s**2 * d[f'dxi_{cor}_err'],
                   fmt='s-', ms=3, capsize=2, color='C0', label=f'{lab} de-trended')
        a.set_xlabel(r'$s\ [h^{-1}{\rm Mpc}]$'); a.legend()
    ax[0].set_ylabel(r'$s^2\,\Delta\xi$')
    fig.tight_layout(); save(fig, 'detrend_effect')


def fig_kaiser():
    """Kaiser consistency (Section 6.4): a single amplitude fitted to
    Delta-xi_0 over 60 < s < 140 Mpc/h predicts Delta-xi_2 with no remaining
    freedom, through the fingers-of-god model at the fiducial (b, k*, sigma_v).
    """
    ch = chain(FIDUCIAL)
    b, kst, sv = median(ch, 'Bnt'), median(ch, 'kstar'), median(ch, 'sigv')
    G = np.load(os.path.join(D, 'templates', 'templates_fog.npz'))
    f = float(G['growth_f'])
    j, i = np.argmin(np.abs(G['kstars'] - kst)), np.argmin(np.abs(G['sigv'] - sv))
    # a constant Delta-b gives Delta-xi_ell = 2 Db [b A_ell^0 + f A_ell^1]
    sh0 = 2 * (b * G['A0_0'][j, i] + f * G['A0_1'][j, i])
    sh2 = 2 * (b * G['A2_0'][j, i] + f * G['A2_1'][j, i])
    d = np.load(os.path.join(D, 'lrg_multipoles.npz'))
    s = d['s']
    sh0, sh2 = np.interp(s, G['s'], sh0), np.interp(s, G['s'], sh2)
    m = (s > 60) & (s < 140)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    for a, mk in zip(ax, ('lumzc', 'lumw1c')):
        dd = d[f'xi_{mk}'] - d['xi_num']
        e = jk_err(d[f'jk_{mk}'].astype(float) - d['jk_num'])
        Db = np.sum(dd[0][m] * sh0[m] / e[0][m] ** 2) / np.sum((sh0[m] / e[0][m]) ** 2)
        a.axhline(0, color='gray', lw=0.8)
        a.axvspan(140, 200, color='0.85', zorder=0)
        a.axvspan(40, 60, color='0.93', zorder=0)
        a.errorbar(s, s ** 2 * dd[1], s ** 2 * e[1], fmt='ko', ms=4, capsize=2,
                   label=r'$\Delta\xi_2$ data')
        a.plot(s, s ** 2 * Db * sh2, 'r-', lw=1.6,
               label=r'Kaiser + FoG prediction from $\Delta\xi_0$, no free parameters')
        a.set_xlabel(r'$s\ [h^{-1}{\rm Mpc}]$')
        a.set_ylabel(r'$s^2\,\Delta\xi_2$')
        a.text(0.04, 0.06, LB[mk], transform=a.transAxes, fontsize=10)
        a.set_xlim(40, 200)
        a.legend(fontsize=9)
    fig.tight_layout(); save(fig, 'kaiser_consistency')


def fig_hod():
    """Halo-occupation null test (Appendix B): Delta-xi of the data against
    the mean of the AbacusSummit boxes with a luminosity-dependent occupation
    and with mass-independent marks (B_Delta = 0 in both by construction)."""
    d = np.load(os.path.join(D, 'lrg_multipoles.npz'))
    s = d['s']
    dx = d['xi_lumw1c'] - d['xi_num']
    err = jk_err(d['jk_lumw1c'].astype(float) - d['jk_num'])
    boxes = sorted(f[:-4] for f in os.listdir(os.path.join(D, 'hod'))
                   if f.startswith('c000_ph') and not f.endswith('_ctrl.npz'))
    load = lambda n: np.load(os.path.join(D, 'hod', n + '.npz'))
    sig = np.array([load(b)['xil'] - load(b)['xin'] for b in boxes])
    ctl = np.array([load(b + '_ctrl')['xil'] - load(b + '_ctrl')['xin'] for b in boxes])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for li, a in enumerate(axes):
        k = s <= (200 if li == 0 else 140)
        a.axhline(0, color='gray', lw=0.8)
        a.errorbar(s[k], s[k] ** 2 * dx[li][k], s[k] ** 2 * err[li][k], fmt='ko', ms=3.5,
                   capsize=2, label='DR1 LRG, W1 mark')
        a.plot(s[k], s[k] ** 2 * sig[:, li].mean(0)[k], color='C3', lw=1.6,
               label='HOD, luminosity-dependent occupation')
        a.plot(s[k], s[k] ** 2 * ctl[:, li].mean(0)[k], color='C0', lw=1.2, ls='--',
               label='HOD, mass-independent marks')
        a.set_ylabel(rf'$s^2\Delta\xi_{2 * li}\ [h^{{-2}}{{\rm Mpc}}^2]$')
        a.set_xlabel(r'$s\ [h^{-1}{\rm Mpc}]$')
        a.set_xlim(40, 200 if li == 0 else 140)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); save(fig, 'hod_null')


def fig_leak():
    """Window and estimator effects (Appendix D): real-space log-normal mocks,
    in which the true xi_2 and xi_4 vanish, against the DR1 statistical error."""
    t = np.load(os.path.join(D, 'leakage_test.npz'))
    LN = np.load(os.path.join(D, 'leakage_mocks.npz'))['xi']
    s, N = t['s'], len(LN)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for a, (li, ell) in zip(axes, ((1, 2), (2, 4))):
        a.axhline(0, color='gray', lw=0.8)
        a.fill_between(s, -s ** 2 * t['ezstd'][li], s ** 2 * t['ezstd'][li], color='C0',
                       alpha=0.18, lw=0, label=r'$\pm1\sigma$, DR1 (EZmocks)')
        a.plot(s, s ** 2 * LN[0, li], color='0.45', lw=1, label='one realisation')
        a.errorbar(s, s ** 2 * t['lnmean'][li], s ** 2 * t['lnstd'][li] / np.sqrt(N),
                   fmt='ko', ms=3.5, capsize=2, label=f'mean of {N} realisations')
        a.set_ylabel(rf'$s^2\xi_{ell}\ [h^{{-2}}{{\rm Mpc}}^2]$')
        a.set_xlabel(r'$s\ [h^{-1}{\rm Mpc}]$')
        a.set_xlim(40, 200)
    axes[0].legend(fontsize=8, loc='upper left')
    fig.tight_layout(); save(fig, 'leakage_mocks')


def fig_fit():
    """Fiducial fit (Fig. 3): xi_0 and xi_2 of both fields and Delta xi_0, with
    the best-fitting sample of the fiducial chain."""
    from likelihood import Likelihood
    like = Likelihood()
    ch = chain(FIDUCIAL)
    top = np.arange(len(ch['samples']))[-500:]     # dead points rise in likelihood
    chi2 = [like.chi2(ch['samples'][i]) for i in top]
    best = ch['samples'][top[int(np.argmin(chi2))]]
    th = like.theta(best)
    print(f'  best-fitting chi^2 = {min(chi2):.1f} for {like.y.size} points')
    d = np.load(os.path.join(D, 'lrg_multipoles.npz'))
    s = d['s']
    m0 = (s >= 40) & (s <= 200)
    m2 = (s >= 40) & (s <= 140)
    mod = model_fog(th, like.tpl, s[m0], s[m2], like.xi_cip)
    n0, n2 = m0.sum(), m2.sum()
    mn0, mn2 = mod[:n0], mod[n0:n0 + n2]
    ml0, ml2 = mod[n0 + n2:2 * n0 + n2], mod[2 * n0 + n2:]
    en, el = jk_err(d['jk_num']), jk_err(d['jk_lumw1c'])
    ed = jk_err(d['jk_lumw1c'] - d['jk_num'])
    cn, cl = '#222222', '#e66101'

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for a, ell, msk, mn, ml in ((ax[0], 0, m0, mn0, ml0), (ax[1], 1, m2, mn2, ml2)):
        x = s[msk]
        a.errorbar(x - 0.6, x ** 2 * d['xi_num'][ell][msk], x ** 2 * en[ell][msk],
                   fmt='o', ms=3.5, lw=0.9, capsize=0, color=cn, label='number weighted')
        a.errorbar(x + 0.6, x ** 2 * d['xi_lumw1c'][ell][msk], x ** 2 * el[ell][msk],
                   fmt='s', ms=3.5, lw=0.9, capsize=0, color=cl,
                   label='luminosity weighted (W1)')
        a.plot(x, x ** 2 * mn, '-', lw=1.4, color=cn)
        a.plot(x, x ** 2 * ml, '-', lw=1.4, color=cl)
        a.set_ylabel(rf'$s^2\,\xi_{2 * ell}\ [h^{{-2}}{{\rm Mpc}}^2]$')
    ax[0].legend(frameon=False, fontsize=9)
    x = s[m0]
    dd = d['xi_lumw1c'][0] - d['xi_num'][0]
    ax[2].axhline(0, color='0.8', lw=0.7)
    ax[2].errorbar(x, x ** 2 * dd[m0], x ** 2 * ed[0][m0], fmt='s', ms=3.5, lw=0.9,
                   capsize=0, color=cl)
    ax[2].plot(x, x ** 2 * (ml0 - mn0), '-', lw=1.4, color=cl)
    ax[2].set_ylabel(r'$s^2\,\Delta\xi_0\ [h^{-2}{\rm Mpc}^2]$')
    for a, hi in zip(ax, (205, 145, 205)):
        a.set_xlabel(r'$s\ [h^{-1}{\rm Mpc}]$')
        a.set_xlim(35, hi)
    fig.tight_layout()
    save(fig, 'fit_overlay')


def fig_corner(tag=FIDUCIAL):
    """Weighted posterior of one fit (Appendix F).

    Shows the sampled parameters listed in 'active'. For the reparametrised
    fits they are taken from 'derived' in their physical form: the sum and
    difference fit shows B_n,Delta and B_L,Delta, and the CIP-product fit
    shows B_L,Delta^2 B_CIP in place of v. Axis ranges are the weighted
    0.1-99.9 per cent intervals.
    """
    try:
        import corner
    except ImportError:
        print('  (skipping corner: pip install corner)'); return
    d = chain(tag)
    nm, act = list(d['names']), list(d['active'])
    x = d['derived'][:, act].copy() if 'derived' in d.files else d['samples']
    lab = [TEX.get(nm[i], nm[i]) for i in act]
    if tag.split('_')[2:3] == ['prod']:
        ic, il = act.index(nm.index('Bcip')), nm.index('BLD')
        x[:, ic] = d['derived'][:, il] ** 2 * x[:, ic]
        lab[ic] = r'$B_{\rm L,\Delta}^2 B_{\rm CIP}$'
    w = d['weights'] / d['weights'].sum()
    rng = []
    for k in range(x.shape[1]):
        o = np.argsort(x[:, k])
        rng.append(tuple(np.interp([0.001, 0.999], np.cumsum(w[o]), x[o, k])))
    fig = corner.corner(x, weights=w, labels=lab, range=rng,
                        show_titles=False, plot_datapoints=False,
                        fill_contours=True, levels=(0.68, 0.95), smooth=1.0)
    save(fig, 'corner_baseline' if tag == FIDUCIAL else f'corner_{tag}', dpi=130)


def fig_3pcf():
    """Marked three-point function (Section 5.4).

    The two caps are fitted jointly with a block-diagonal covariance and
    combined for display with one inverse-variance weight per cap.
    """
    d3 = os.path.join(D, '3pcf')

    def load(cap):
        o = {}
        for tag, key in (('w10', 'nnn'), ('w01', 'LLL')):
            m = np.loadtxt(os.path.join(d3, f'LRG_{cap}_{tag}_jk.3pcf'), skiprows=1)
            j = np.loadtxt(os.path.join(d3, f'LRG_{cap}_{tag}_jk.3pcf.jk'), skiprows=2)
            o[key] = m[:, 8]
            o['jk_' + key] = j[:, 9].reshape(len(m), len(j) // len(m))
        o['bins'], o['nreal'] = m[:, :6], o['jk_nnn'].shape[1]
        o['ok'] = np.all([np.isfinite(o[k]) if o[k].ndim == 1
                          else np.isfinite(o[k]).all(axis=1)
                          for k in ('nnn', 'LLL', 'jk_nnn', 'jk_LLL')], axis=0)
        return o

    def jke(a, nr):
        return np.sqrt((nr - 1) / nr * ((a - a.mean(1, keepdims=True)) ** 2).sum(1))

    caps = {c: load(c) for c in ('NGC', 'SGC')}
    num = den = 0.0
    per = {}
    for cap, c in caps.items():
        ok, nr = c['ok'], c['nreal']
        jd = (c['jk_LLL'] - c['jk_nnn'])[ok]
        nb = int(ok.sum())
        C = (nr - 1) / nr * np.cov(jd - jd.mean(1, keepdims=True), bias=True) * nr
        Ci = hartlap(nr, nb) * np.linalg.inv(C)
        dv, tv = (c['LLL'] - c['nnn'])[ok], c['nnn'][ok]
        per[cap] = (float(tv @ Ci @ dv / (tv @ Ci @ tv)),
                    float(1 / np.sqrt(tv @ Ci @ tv)))
        num += tv @ Ci @ dv
        den += tv @ Ci @ tv
    A, sA = float(num / den), float(1 / np.sqrt(den))

    okc = caps['NGC']['ok'] & caps['SGC']['ok']
    sb = caps['NGC']['bins'].reshape(-1, 3, 2).mean(axis=2)
    order = np.lexsort((sb[okc][:, 2], sb[okc][:, 1], sb[okc][:, 0]))
    x = np.arange(int(okc.sum()))
    w, val, var = {}, {}, {}
    for cap, c in caps.items():
        e_n = jke(c['jk_nnn'][okc], c['nreal'])[order]
        w[cap] = 1.0 / e_n ** 2
        val[cap] = {'nnn': c['nnn'][okc][order], 'LLL': c['LLL'][okc][order],
                    'd': (c['LLL'] - c['nnn'])[okc][order]}
        var[cap] = {'nnn': e_n ** 2,
                    'd': jke((c['jk_LLL'] - c['jk_nnn'])[okc], c['nreal'])[order] ** 2}
    W = sum(w.values())
    cb = {k: sum(w[c] * val[c][k] for c in caps) / W for k in ('nnn', 'LLL', 'd')}
    er = {k: np.sqrt(sum(w[c] ** 2 * var[c][k] for c in caps)) / W
          for k in ('nnn', 'd')}

    fig, ax = plt.subplots(2, 1, figsize=(11, 6.3), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 1.2], hspace=0.08))
    ax[0].axhline(0, color='0.7', lw=0.8)
    ax[0].errorbar(x, cb['nnn'], yerr=er['nnn'], fmt='o', ms=3.4, lw=0.9,
                   color='#222222', capsize=0, label=r'$\zeta_{\rm nnn}$')
    ax[0].plot(x, cb['LLL'], '-', lw=1.2, color='#ca0020',
               label=r'$\zeta_{\rm LLL}$')
    ax[0].set_yscale('symlog', linthresh=1e-4)
    ax[0].set_ylabel(r'$\zeta$')
    ax[0].legend(frameon=False, fontsize=9, loc='upper right')
    ax[1].axhline(0, color='0.7', lw=0.8)
    ax[1].fill_between(x, (A - sA) * cb['nnn'], (A + sA) * cb['nnn'],
                       color='#0571b0', alpha=0.28, lw=0)
    ax[1].plot(x, A * cb['nnn'], '-', lw=1.4, color='#0571b0',
               label=rf'$A\,\zeta_{{\rm nnn}}$, $A={A:.3f}\pm{sA:.3f}$')
    ax[1].errorbar(x, cb['d'], yerr=er['d'], fmt='o', ms=3.4, lw=0.9,
                   color='#ca0020', capsize=0, label=r'$\Delta\zeta$ (NGC+SGC)')
    ax[1].set_yscale('symlog', linthresh=1e-5)
    ax[1].set_ylim(bottom=-4e-2)
    ax[1].set_ylabel(r'$\Delta\zeta$')
    ax[1].legend(frameon=False, fontsize=9, loc='upper right')
    ins = ax[1].inset_axes([0.055, 0.055, 0.30, 0.235])
    for k, (lab, a_, e_) in enumerate([('NGC', *per['NGC']), ('SGC', *per['SGC']),
                                       ('joint', A, sA)]):
        ins.errorbar(a_, -k, xerr=e_, fmt='o', ms=3.4, lw=1.1, capsize=2.0,
                     color='#ca0020' if lab == 'joint' else '#222222')
    ins.set_yticks([0, -1, -2]); ins.set_yticklabels(['NGC', 'SGC', 'joint'], fontsize=7.5)
    ins.set_ylim(-2.7, 1.05); ins.set_xlim(0.085, 0.185)
    ins.tick_params(labelsize=7.5, length=2.5, pad=1.5)
    ins.set_xlabel(r'$A$', fontsize=8, labelpad=0.5)
    ins.set_zorder(5); ins.patch.set_alpha(1.0)
    ax[1].set_xlabel(r'triangle configuration, ordered by $(r_1, r_2, r_3)$')
    ax[1].set_xlim(-0.5, len(x) - 0.5)
    save(fig, 'zeta_3pcf')


def save(fig, name, dpi=150):
    p = os.path.join(F, name + '.png')
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    print(f'  wrote {os.path.relpath(p)}')


ALL = {'data': fig_data, 'dxi': fig_dxi, 'fit': fig_fit, '3pcf': fig_3pcf,
       'detrend': fig_detrend, 'kaiser': fig_kaiser, 'hod': fig_hod,
       'leak': fig_leak, 'corner': fig_corner}
if __name__ == '__main__':
    if {'-h', '--help'} & set(sys.argv[1:]):
        print(__doc__.strip()); sys.exit()
    want = sys.argv[1:] or list(ALL)
    i = 0
    while i < len(want):
        k, i = want[i], i + 1
        if k == 'corner':
            # chain tags given after 'corner' replace the baseline
            tags = []
            while i < len(want) and want[i] not in ALL:
                tags.append(want[i]); i += 1
            for t in tags or [FIDUCIAL]:
                if not os.path.exists(os.path.join(D, 'chains', f'chain_{t}.npz')):
                    print(f'  no chain "{t}" in data/chains'); continue
                fig_corner(t)
        elif k not in ALL:
            print(f'  unknown figure "{k}"; choose from {list(ALL)}')
        else:
            ALL[k]()
