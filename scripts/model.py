"""Model for the luminosity-weighted correlation function.

Implements the multipole model of Section 2 of the paper: the Barkana & Loeb
(2011) relative clustering model as extended by Soumagnac et al. (2016, 2019),
in redshift space (Kaiser) for the monopole and quadrupole. model_fog adds the
fingers-of-god damping of the fiducial fit.

    xi_0,X = (B_Xt^2 + 2/3 f B_Xt + f^2/5) xi_tot0
             + (2 B_Xt + 2f/3) B_XD xi_add0 + B_XD^2 B_CIP xihat_CIP + Bsys0_X
    xi_2,X = (4/3 f B_Xt + 4/7 f^2) xi_tot2
             + (4f/3) B_XD xi_add2 + Bsys2_X

with X in {n, L}. The CIP term enters the monopole only.

B_CIP appears only through the product B_XD^2 B_CIP, so the posterior has a
funnel shape (see Appendix E of the paper).
"""
import os

import numpy as np

PARAMS = ['Bnt', 'BLt', 'BnD', 'BLD', 'Bcip',
          'Bsys_n', 'Bsys_L', 'Bsys2_n', 'Bsys2_L', 'Amc', 'kstar']

# The fiducial fit adds a fingers-of-god dispersion sigma_v and fits the
# quadrupole to 140 Mpc/h. It uses templates_fog.npz, in which the linear
# Kaiser coefficients are replaced by tables A_l^(n), B_l^(n) on a grid of
# (k*, sigma_v). Fiducial chain: chain_mp_lumw1c_nocip_q140_fog.
PARAMS_FOG = PARAMS + ['sigv']


class Templates:
    """CAMB-derived templates on a grid of the BAO damping scale k*."""

    def __init__(self, path):
        with np.load(path) as z:            # read once; the file is compressed
            self.t = {k: z[k] for k in z.files}
        self.ks = self.t['kstars']
        self.s = self.t['s']
        self.f = float(self.t['growth_f'])
        self.zeff = float(self.t['zeff'])

    def _interp(self, tab, kstar):
        j = np.clip(np.searchsorted(self.ks, kstar) - 1, 0, len(self.ks) - 2)
        w = (kstar - self.ks[j]) / (self.ks[j + 1] - self.ks[j])
        return (1 - w) * tab[j] + w * tab[j + 1]

    def pieces(self, kstar, amc, s_out):
        xt0 = self._interp(self.t['xi_tot'], kstar)
        xa0 = self._interp(self.t['xi_add'], kstar)
        x1 = self._interp(self.t['xi1'], kstar)
        xt0 = xt0 + amc * np.gradient(xt0, self.s) * x1     # mode coupling
        xt2 = -self._interp(self.t['xi_tot2'], kstar)       # i^2 sign
        xa2 = -self._interp(self.t['xi_add2'], kstar)
        out = [np.interp(s_out, self.s, y)
               for y in (xt0, xa0, xt2, xa2, self.t['xi_cip'])]
        return out


PARAMS_MONO = ['Bnt', 'BLt', 'BnD', 'BLD', 'Bcip',
               'Bsys_n', 'Bsys_L', 'Amc', 'kstar']


def model(theta, tpl, s_out, multipoles=True):
    """Return the stacked data vector [xi_n, xi_L], each [xi0] or [xi0, xi2].

    Accepts either the 11-parameter multipole vector or the 9-parameter
    monopole vector (which omits the two quadrupole offsets)."""
    if len(theta) == 9:
        bnt, blt, bnd, bld, bcip, bsn, bsl, amc, kst = theta
        bs2n = bs2l = 0.0
    else:
        (bnt, blt, bnd, bld, bcip,
         bsn, bsl, bs2n, bs2l, amc, kst) = theta
    f = tpl.f
    xt0, xa0, xt2, xa2, xc = tpl.pieces(kst, amc, s_out)
    out = []
    for bt, bd, b0, b2 in ((bnt, bnd, bsn, bs2n), (blt, bld, bsl, bs2l)):
        if multipoles:
            m0 = ((bt**2 + 2 * f * bt / 3 + f**2 / 5) * xt0
                  + (2 * bt + 2 * f / 3) * bd * xa0 + bd**2 * bcip * xc + b0)
            m2 = (4 * f * bt / 3 + 4 * f**2 / 7) * xt2 + (4 * f / 3) * bd * xa2 + b2
            out.append(np.concatenate([m0, m2]))
        else:
            out.append(bt**2 * xt0 + 2 * bt * bd * xa0 + bd**2 * bcip * xc + b0)
    return np.concatenate(out)


def hartlap(n_real, n_bins):
    """Debias an inverse covariance estimated from n_real realisations."""
    return (n_real - n_bins - 2) / (n_real - 1)


def percival(n_real, n_bins, n_par):
    """Percival et al. (2014) factor m1 for the parameter covariance."""
    d1, d4, d2 = n_real - n_bins - 1, n_real - n_bins - 4, n_real - n_bins - 2
    A, B = 2.0 / (d1 * d4), d2 / (d1 * d4)
    return (1 + B * (n_bins - n_par)) / (1 + A + B * (n_par + 1))


class TemplatesFoG:
    """A_l^(n), B_l^(n) tables on the (k*, sigma_v) grid of the fiducial fit.

    With the damping D = [1 + (k mu sigma_v)^2 / 2]^-1 the mu integrals have no
    closed form, so they are tabulated and interpolated bilinearly.
    """

    def __init__(self, path):
        with np.load(path) as z:            # read once; the file is compressed
            self.t = {k: z[k] for k in z.files}
        self.ks, self.sv = self.t['kstars'], self.t['sigv']
        self.s = self.t['s']
        self.f = float(self.t['growth_f'])

    def _interp2(self, tab, kstar, sigv):
        j = np.clip(np.searchsorted(self.ks, kstar) - 1, 0, len(self.ks) - 2)
        i = np.clip(np.searchsorted(self.sv, sigv) - 1, 0, len(self.sv) - 2)
        u = (kstar - self.ks[j]) / (self.ks[j + 1] - self.ks[j])
        v = (sigv - self.sv[i]) / (self.sv[i + 1] - self.sv[i])
        return ((1 - u) * (1 - v) * tab[j, i] + u * (1 - v) * tab[j + 1, i]
                + (1 - u) * v * tab[j, i + 1] + u * v * tab[j + 1, i + 1])

    def AB(self, kstar, sigv):
        A = {(l, n): self._interp2(self.t[f'A{l}_{n}'], kstar, sigv)
             for l in (0, 2) for n in (0, 1, 2)}
        B = {(l, n): self._interp2(self.t[f'B{l}_{n}'], kstar, sigv)
             for l in (0, 2) for n in (0, 1)}
        return A, B

    def xi1(self, kstar):
        ks = self.ks
        j = np.clip(np.searchsorted(ks, kstar) - 1, 0, len(ks) - 2)
        w = (kstar - ks[j]) / (ks[j + 1] - ks[j])
        return (1 - w) * self.t['xi1'][j] + w * self.t['xi1'][j + 1]


def model_fog(theta, tpl, s0, s2, xi_cip):
    """Fiducial model: stacked [xi_n(s0), xi_n(s2), xi_L(s0), xi_L(s2)].

    s0 and s2 are separate because the fiducial cuts the quadrupole at
    140 Mpc/h while keeping the monopole to 200.
    """
    (bnt, blt, bnd, bld, bcip, bsn, bsl,
     bs2n, bs2l, amc, kst, sv) = theta[:12]
    f = tpl.f
    A, B = tpl.AB(kst, sv)
    x1 = tpl.xi1(kst)
    out = []
    for bt, bd, b0, b2 in ((bnt, bnd, bsn, bs2n), (blt, bld, bsl, bs2l)):
        t0 = bt**2 * A[(0, 0)] + 2 * bt * f * A[(0, 1)] + f**2 * A[(0, 2)]
        t0 = t0 + amc * np.gradient(t0, tpl.s) * x1
        m0 = t0 + 2 * bd * (bt * B[(0, 0)] + f * B[(0, 1)]) + b0
        m2 = (bt**2 * A[(2, 0)] + 2 * bt * f * A[(2, 1)] + f**2 * A[(2, 2)]
              + 2 * bd * (bt * B[(2, 0)] + f * B[(2, 1)]) + b2)
        out.append(np.concatenate([np.interp(s0, tpl.s, m0) + bd**2 * bcip
                                   * np.interp(s0, tpl.s, xi_cip),
                                   np.interp(s2, tpl.s, m2)]))
    return np.concatenate(out)
