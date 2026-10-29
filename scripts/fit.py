"""Run the multipole fits of the paper from the shipped data vectors.

    python fit.py                          fiducial fit (B_CIP = 0)
    python fit.py --fix-bld                null model, B_L,Delta = 0
    python fit.py --mockcal [--fix-bld]    the same with the likelihood
                                           calibrated on the EZmocks
    python fit.py --ratio 1.017            B_L,Delta = 1.017 B_n,Delta
    python fit.py --sumdiff                common mode S and difference D
    python fit.py --product                CIP limit, sampling v = B_CIP (B_n^2 + B_L^2)
    python fit.py --cip-positive           B_CIP free, 0 <= B_CIP <= 0.3
    python fit.py --mark inj3 --num ninj3  fits to the injected vectors
                                           (lrg_injected.npz); --bldrange 30
                                           widens the amplitude priors

The quoted B_L,Delta interval comes from the fiducial fit, widened as in
compute_results.py; the quoted Bayes factor is the difference of ln Z between
the two --mockcal runs.

Sampler settings are those of the paper's fits. Fits with B_CIP = 0 use
dynesty's static nested sampler with 500 live points, multi-ellipsoid bounds
and random-walk proposals (dynesty's default of ndim + 20 walks), stopping at
d ln Z < 0.1. Fits with B_CIP free (--product, --cip-positive), and with
--dynamic, use the dynamic sampler with random-direction slice proposals
(rslice), 500 initial live points and dynesty's default stopping rule. The
random generator is seeded with 3. A static run takes about four minutes on
one core and the dynamic CIP-product fit about twenty.

Output: <outdir>/chain_<tag>.npz with the same keys and tag as data/chains,
so compute_results.py --chains <outdir> uses it in place of the shipped one.
"""
import argparse
import os
import sys
import time

import numpy as np
import dynesty

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from likelihood import IBL, NAMES, Likelihood, calibration_factor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mark', default='lumw1c',
                help='luminosity-weighted field: lumw1c, lumzc, mstarfs, lumw1, '
                     'lumz, or an injected one (inj3, inj5, inj10, inj20, inj30)')
ap.add_argument('--num', default='num',
                help='number-weighted field: num, or ninj3, ninj10')
ap.add_argument('--fix-bld', action='store_true', help='fix B_L,Delta = 0')
ap.add_argument('--mockcal', action='store_true',
                help='divide the inverse covariance by r^2 (Appendix B)')
ap.add_argument('--calibration', default=None, metavar='FILE',
                help='with --mockcal, take r from FILE (written by '
                     'calibrate_covariance.py) instead of data/ezmock/jk_covtest.npz')
ap.add_argument('--ratio', type=float, default=None,
                help='impose B_L,Delta = RATIO * B_n,Delta (1.017 in the paper)')
ap.add_argument('--sumdiff', action='store_true',
                help='sample S = (B_L,Delta + B_n,Delta)/2 and D = B_L,Delta - B_n,Delta')
ap.add_argument('--product', action='store_true',
                help='B_CIP free, sampling v = B_CIP (B_n,Delta^2 + B_L,Delta^2) '
                     'in [0, 0.05]')
ap.add_argument('--cip-positive', action='store_true', help='B_CIP free in [0, 0.3]')
ap.add_argument('--bldrange', type=float, default=10.0,
                help='half-width of the priors on B_n,Delta and B_L,Delta')
ap.add_argument('--dynamic', action='store_true',
                help='dynamic sampling with random-direction slice proposals (implied by '
                     '--product and --cip-positive)')
ap.add_argument('--nlive', type=int, default=500,
                help='live points; initial live points for dynamic runs')
ap.add_argument('--dlogz', type=float, default=0.1,
                help='stopping criterion of static runs')
ap.add_argument('--seed', type=int, default=3, help='seed of numpy.random.default_rng')
ap.add_argument('--outdir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 '..', 'runs'),
                help='output directory, default runs/ in the package')
args = ap.parse_args()

like = Likelihood(args.mark, args.num, fix_bld=args.fix_bld, mockcal=args.mockcal,
                  r=calibration_factor(args.calibration) if args.calibration else None,
                  ratio=args.ratio, sumdiff=args.sumdiff,
                  cip_positive=args.cip_positive, product=args.product,
                  bldrange=args.bldrange)
free_cip = args.product or args.cip_positive
tag = (f'mp_{args.mark}' + ('' if args.num == 'num' else f'_{args.num}')
       + ('_nobld' if args.fix_bld else '') + ('' if free_cip else '_nocip')
       + ('_cippos' if args.cip_positive else '')
       + ('_ratio' if args.ratio is not None else '') + ('_prod' if args.product else '')
       + ('_sumdiff' if args.sumdiff else '') + ('_mockcal' if args.mockcal else '')
       + '_q140_fog' + ('' if args.bldrange == 10 else f'_w{args.bldrange:g}'))
print(f'{tag}: {like.y.size} data points, {like.nreal} jackknife realisations, '
      f'{like.ndim} free parameters' + (f', r = {like.r:.4f}' if args.mockcal else ''),
      flush=True)

t0 = time.time()
rng = np.random.default_rng(args.seed)
if free_cip or args.dynamic:
    sampler = dynesty.DynamicNestedSampler(like, like.prior_transform, like.ndim,
                                           bound='multi', sample='rslice', rstate=rng)
    sampler.run_nested(nlive_init=args.nlive, print_progress=False)
else:
    sampler = dynesty.NestedSampler(like, like.prior_transform, like.ndim,
                                    nlive=args.nlive, bound='multi', sample='rwalk',
                                    rstate=rng)
    sampler.run_nested(dlogz=args.dlogz, print_progress=False)
res = sampler.results
w = res.importance_weights()
extra = {}
if args.ratio is not None or args.sumdiff or args.product:
    extra['derived'] = np.array([like.theta(u) for u in res.samples])

os.makedirs(args.outdir, exist_ok=True)
out = os.path.join(args.outdir, f'chain_{tag}.npz')
np.savez(out, samples=res.samples, weights=w, names=NAMES, active=like.active,
         logz=res.logz[-1], logzerr=res.logzerr[-1], niter=res.niter, ndim=like.ndim,
         prior_lo=like.lo, prior_hi=like.hi, nlive=args.nlive, seed=args.seed,
         dynamic=bool(free_cip or args.dynamic), dynesty_version=dynesty.__version__,
         **extra)
ess = 1.0 / np.sum((w / w.sum()) ** 2)
print(f'ln Z = {res.logz[-1]:.2f} +- {res.logzerr[-1]:.2f}, {res.niter} iterations, '
      f'effective sample size {ess:.0f}, {time.time() - t0:.0f} s')
if not args.fix_bld:
    x = np.array([like.theta(u)[IBL] for u in res.samples])
    o = np.argsort(x)
    q = np.interp([0.16, 0.5, 0.84], np.cumsum(w[o]) / w.sum(), x[o])
    print(f'B_L,Delta = {q[1]:+.2f} +{q[2] - q[1]:.2f} -{q[1] - q[0]:.2f} '
          '(posterior; compute_results.py applies the covariance calibration)')
print(f'wrote {out}')
