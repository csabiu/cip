# Mass versus light in DESI DR1

Data and code for Sabiu, Yoo & Paudel, *Large-scale distribution of mass
versus light in DESI DR1: a luminosity-weighted clustering search for the
relative distribution of baryons and dark matter*.

The repository contains the measured correlation functions with their
jackknife realisations, the model templates, the posterior chains of the fits
reported in the paper and the inputs of the robustness tests, with scripts
that rerun the LRG multipole fits, the covariance calibration and the marked
three-point fit and regenerate the figures.
Reproduction starts from the measured correlation functions: building the
catalogues and counting pairs needs the DESI catalogues and is not part of the
package (see Provenance and Data not included).

    python scripts/compute_results.py        # Table 2 and the main constraints, from the chains
    python scripts/plot_figures.py           # figures, written to figures/
    python scripts/calibrate_covariance.py   # covariance calibration of Appendix B
    python scripts/fit.py [options]          # rerun a fit, written to runs/
    python scripts/fit_3pcf.py               # marked three-point fit of Section 5.4

The scripts need numpy, scipy and matplotlib; `fit.py` also needs dynesty and
the posterior plots need corner. `plot_figures.py` takes figure names as
arguments (data, dxi, fit, 3pcf, detrend, kaiser, hod, leak, corner) and makes
all of them by default; `corner` followed by chain tags (file names in
`data/chains` without `chain_` and `.npz`) draws the posteriors of those fits.
Figures C1 and C2 of the paper need the per-galaxy photometry, which is not
included; the trends they show are in `data/mark_systematics.txt` and
`data/w1_systematics.txt`.

## Rerunning the fits

`fit.py` builds the likelihood of `scripts/likelihood.py` from
`data/lrg_multipoles.npz` and `data/templates/templates_fog.npz`, samples it
with dynesty and writes a chain in the format of `data/chains`.
`compute_results.py --chains runs` then uses the new chains in place of the
shipped ones of the same name.

| Result | Runs |
|---|---|
| B_L,Δ and B_L,Δ − B_n,Δ (Section 5.2, Table 2) | `fit.py` |
| Bayes factor for B_L,Δ | `fit.py --mockcal` and `fit.py --mockcal --fix-bld` |
| B_L,Δ with the selection ratio imposed | `fit.py --ratio 1.017` |
| common mode S and difference D | `fit.py --sumdiff` |
| CIP limit (Section 5.3) | `fit.py --product` |
| other LRG multipole rows of Table 2 | `fit.py --mark lumzc` (or mstarfs, lumw1, lumz), and `--cip-positive` |
| injection fits (Section 6.5, Appendix C5) | `fit.py --mark inj3 --num ninj3`, with `--bldrange 30` for the wider prior |

The monopole, BGS and modelling-variant fits (Section 6.3 and Appendix C3),
the linear-Kaiser fits and the halo-occupation fit are included as chains
only; `fit.py` builds the fiducial multipole likelihood and the variants
listed above.

Likelihood. The data vector is [ξ₀,n, ξ₂,n, ξ₀,L, ξ₂,L] with ξ₀ over
40–200 Mpc/h and ξ₂ over 40–140 Mpc/h, 104 points. The covariance is the
jackknife covariance of the 400 realisations, (N − 1)/N Σ (x_k − x̄)(x_k − x̄)ᵀ,
and its inverse is multiplied by the Hartlap factor (N − p − 2)/(N − 1). The
log-likelihood is −χ²/2 without a normalisation term, so ln Z can be compared
only between fits that share a data vector and covariance. With `--mockcal`
the inverse covariance is also divided by r², with r = 1.247 from the
covariance calibration; the Bayes factors quoted in the paper come from these
runs. The intervals quoted for the LRG multipole fits come from the runs
without `--mockcal`, widened about the median by r = 1.247.
`compute_results.py` does this, rescaling r by the Percival et al. (2014)
factor for fits with a different number of parameters, and widens the
intervals of the other fits by the Percival factor alone.

Priors (uniform):

| Parameter | Prior |
|---|---|
| B_n,t, B_L,t | [0, 5] |
| B_n,Δ, B_L,Δ | [−10, 10] (`--bldrange`) |
| B_CIP | 0; [0, 0.3] with `--cip-positive` |
| v = B_CIP (B_n,Δ² + B_L,Δ²) | [0, 0.05] with `--product`, with B_CIP ≤ 0.3 |
| S, D | [−30, 30], [−10, 10] with `--sumdiff` |
| B_sys,n, B_sys2,n | [−0.0015, 0.0015] |
| B_sys,L, B_sys2,L | [−0.01, 0.01] |
| A_MC | [0, 6] |
| k* | [0, 10] Mpc/h |
| σ_v | [0, 12] Mpc/h |

Sampler. Fits with B_CIP = 0 use dynesty's static nested sampler with 500
live points, multi-ellipsoid bounds and random-walk proposals (dynesty's
default of ndim + 20 walks), stopping at d ln Z < 0.1; the monopole fits, with
fewer than ten parameters, used uniform sampling within the bounds, which
dynesty 3.1 chooses automatically. Fits with B_CIP free use the dynamic
sampler with random-direction slice proposals, 500 initial live points and
dynesty's default stopping rule, as do the A = 3, 5 and 30 single-field
injection fits (`--dynamic`). The exception is the static comparison run of
Appendix E (`_static`). The random generator is
`numpy.random.default_rng(3)`. A static run takes about four minutes on one
core and the dynamic CIP-product fit about twenty.

Reruns agree with the shipped chains statistically rather than bit for bit.
The shipped jackknife realisations and templates are stored in single
precision, which changes ln L by about 10⁻⁴, and a seeded nested-sampling run
does not follow the same path once any likelihood value differs; the BLAS
build and the dynesty version have the same effect. Rerunning the fits with
these scripts gave:

| Result | Shipped chains | Rerun |
|---|---|---|
| B_L,Δ, fiducial | +0.63 +5.36 −5.35 | +0.50 +5.55 −5.07 |
| B_L,Δ − B_n,Δ | +1.97 ± 1.34 | +1.95 ± 1.32 |
| B_L,Δ, ratio imposed | −2.65 +4.93 −4.62 | −2.82 +4.88 −4.21 |
| S | −0.26 ± 5.35 | −0.39 ± 5.33 |
| D | +2.01 ± 1.32 | +2.03 ± 1.35 |
| ln Z, fiducial and null, `--mockcal` | −58.06, −57.47 | −58.08, −57.71 |
| ln B for B_L,Δ | −0.59 ± 0.34 | −0.38 ± 0.34 |
| r (`calibrate_covariance.py`) | 1.2474 | 1.2474 |
| B_L,Δ² B_CIP, 95 per cent (`--product`) | < 0.013 | < 0.013 |
| A of the marked three-point function (`fit_3pcf.py`) | 0.132 ± 0.012 | 0.132 ± 0.012 |

Medians agreed to within 0.2, interval widths to within 10 per cent and ln Z
to within 0.4, which is the scatter expected between nested-sampling runs.

## Covariance calibration

`calibrate_covariance.py` repeats the calibration of Appendix B from
`data/ezmock/mock_multipoles.npz` (the fitted multipoles of 1000 DR1
EZmocks) and `data/ezmock/mock_jackknife.npz` (the 400-region jackknife of 25
of them). Each mock jackknife covariance is used in turn as the covariance of a
fit to the mocks, linearised about the posterior median of the fiducial fit,
and the scatter of the fitted parameters over the other 999 mocks is compared
with their quoted errors. It prints the ratios of every parameter with and
without the Hartlap and Percival factors, r = 1.247 for B_L,Δ, the noise
factor h and the half-split test, in about ten seconds. The scripts can read
its output in place of the shipped `data/ezmock/jk_covtest.npz`:

    python scripts/calibrate_covariance.py --out runs/calibration.npz
    python scripts/fit.py --mockcal --calibration runs/calibration.npz
    python scripts/compute_results.py --calibration runs/calibration.npz

The effective-number and mark-variance diagnostics in `jk_covtest.npz` need
the galaxy and mock catalogues and are not recomputed.

## Correlation functions

Landy–Szalay estimates for the two Galactic caps combined, in 32 bins of
5 Mpc/h over 40 < s < 200 Mpc/h, with the delete-one jackknife realisations.
The covariance of Δξ = ξ_L − ξ_n is built from the per-realisation
differences, so that the cosmic variance common to the two weightings cancels.

- `lrg_multipoles.npz`: the multipoles used in the fits. `xi_<field>` has shape
  (2, 32) for ℓ = 0, 2 and `jk_<field>` has shape (400, 2, 32). Fields are
  `num` and the marks `lumw1c` (baseline), `lumzc`, `mstarfs`, `lumw1` and
  `lumz`. The fits use ξ₀ over 40–200 Mpc/h and ξ₂ over 40–140 Mpc/h.
- `lrg_monopole.npz`, `bgs_monopole.npz`, `bgsf_monopole.npz`: monopoles of the
  LRG, BGS_BRIGHT-21.5 and faint BGS samples, with Δξ, its covariance and the
  jackknife realisations (256 regions for the LRGs, 128 for BGS). The BGS fits
  use 40–150 Mpc/h.
- `lrg_injected.npz`: multipoles with an injected signal (Section 6.5 and
  Appendix C5). `inj<A>` has the W1 mark multiplied by 1 + A δ_r.
  `ninj<A>` is the number-weighted field with the weights multiplied by
  1 + (A/R) δ_r, with R = 1.017 the ratio of the two amplitudes set by the
  selection; fits use it with `inj<A>`.
  `ln1real*` is a real-space log-normal mock without and with an injection.
- `lrg_cross.npz`: the cross-correlation of the number- and luminosity-weighted
  fields (Appendix C6).

The marks `lumw1c` and `lumzc` are the W1 and z-band luminosity marks after
de-trending against the imaging maps, `lumw1` and `lumz` the same before
de-trending, `mstarfs` the FastSpecFit stellar mass and `lumr` the r-band
mark of the BGS samples.

## Templates

`data/templates/` holds the CAMB templates at the effective redshift of each
sample, on a grid of the damping scale k*: `templates.npz` for the LRGs and
`templates_bgs.npz`, `templates_bgsf.npz` for the two BGS samples.
`templates_fog.npz` and `templates_fog_bgs.npz` add the fingers-of-god
dispersion σ_v as a second grid axis; the fiducial fit uses
`templates_fog.npz`. `templates_desi.npz` is the DESI-like damping template of
Appendix C3. `scripts/model.py` evaluates the model from these files.

## Chains

`data/chains/` holds the dynesty chains of the 53 fits reported in the paper.
Each file has `samples`, `weights`, `names`, `active` (the indices of `names`
that were sampled), `logz`, `logzerr` and `ndim`, and `derived` for the
reparametrised fits. The samples have to be used with their weights. In the
reparametrised fits the sampled columns keep the names of the slots they
occupy: in the `sumdiff` fits the `BnD` and `BLD` columns of `samples` hold S
and D, and in the `prod` fit the `Bcip` column holds v; `derived` gives the
physical parameters.

The chain tags (the file names without `chain_` and `.npz`) give the fit
settings. `mp` is the joint monopole and quadrupole fit and `mono` the
monopole alone, followed by the mark. `nocip` fixes B_CIP = 0, `cippos`
restricts it to B_CIP ≥ 0 and `nobld` fixes B_L,Δ = 0. `ratio` is the fit
with B_L,Δ/B_n,Δ fixed by the selection, `sumdiff` the fit in the sum and
difference of the two amplitudes and `prod` the CIP fit sampled in
v = B_CIP (B_n,Δ² + B_L,Δ²). `_q140` cuts the quadrupole at
140 Mpc/h, `_fog` includes the fingers-of-god dispersion, `_mockcal` uses the
likelihood calibrated on the EZmocks and `_w30` a prior U[−30, 30] on the
scale-dependent amplitudes. `inj` and `ninj` are fits to the injected data and
`hodmean5` the fit to the mean of the halo-occupation boxes. The prefixes
`BGS_` and `BGSF_` mark the two BGS samples, and `_fog_sv0` fixes σ_v = 0.
The other suffixes are the scale-cut and modelling variants of Section 6.3 and
Appendix C3. The fiducial fit is `chain_mp_lumw1c_nocip_q140_fog.npz`.

`table2_fitstats.npz` gives χ²_min and the numbers of data points and
parameters for the rows of Table 2.

## Robustness tests

- `shuffle/`: the mark-shuffle ensembles of Section 6.1 (400 permutations of
  each mark, and 100 against a 1:1 subsample of the randoms) and their summary.
- `hod/`: the halo-occupation test of Appendix B. One file per AbacusSummit
  box with a luminosity-dependent occupation (`c000_ph*.npz`) and one with
  mass-independent marks (`*_ctrl.npz`), the mean of the boxes
  (`hodmean5.npz`) and the paired estimate of the leakage (`hod_paired.npz`).
- `ezmock/`: the covariance calibration of Appendix B: the multipoles of the
  1000 DR1 EZmocks, the jackknife repeated on 25 of them, and the results.
- `leakage_mocks.npz`, `leakage_test.npz`: the window test of Appendix D, with
  ξ₀, ξ₂ and ξ₄ of each real-space log-normal mock.
- `3pcf/`: the marked three-point function of Section 5.4. All the GRAMSCI
  runs with their jackknife realisations (`.jk`), the difference
  `delta_zeta_jk.npz` and the fit `zeta_caps.npz`, which `fit_3pcf.py`
  recomputes. `w10` is unweighted and `w01` luminosity weighted; the NGC `w11`
  and `w12` runs are not used in the paper.
- `band_alpha.npz` (Appendix C2), `kaiser_consistency.npz` (Section 6.4),
  `injection_error.npz` and `inject_full_prep.npz` (Appendix C5),
  `appendix_cov.npz` (Appendix B),
  `forecast_y5.npz` and `forecast_3pcf.npz` (Sections 7 and 5.4),
  `response_pred.npz` (Appendix A2) and `samples.npz` (Fig. 1).

## Notes on the injection tests

The injection tests of Appendix C5 multiply each mark by 1 + A δ_r(x), where
δ_r = [r(k) − r_lss] δ_m is the matter field filtered by the relative-clustering
kernel, so that A is added to B_L,Δ. The construction is as follows.

1. The mark is modulated by the filtered field. Modulating it by the
   local density changes only the scale-independent part of Δξ and leaves
   ξ_add unchanged.
2. The gridded galaxy field is in redshift space, (b + f μ²) δ_m. Dividing by b
   instead of inverting the Kaiser operator along the local line of sight
   leaves a factor (1 + β μ²), which biases the monopole by about 15 per cent
   and the quadrupole by a factor of about 2.4. δ_r itself has no Kaiser
   factor.
3. The overdensity is measured against the randoms; against a global mean
   the radial selection is imprinted as a signal.
4. The bias that normalises the field is taken from the Kaiser monopole,
   (b² + 2fb/3 + f²/5) ξ, not from b² ξ.
5. Shot noise in the galaxy field passes through the filter and adds a term to
   Δξ with the same BAO features as the signal. Deconvolving the
   cloud-in-cell window amplifies it. We clip δ_r at 10σ.
6. A physical signal changes both amplitudes in the ratio set by the selection,
   so the number weights are modulated as well (Section 6.5).
7. The injections above modulate the finished marks. Modulating the W1 fluxes
   instead and repeating the redshift normalisation, de-trending and clipping
   leaves 99.9 ± 0.2 per cent of the injected change in ξ_L at A = 10
   (Appendix C5, `inject_full_prep.npz`). They differ mainly by a 0.1 per
   cent rescaling of the uninjected ξ_L; fitting the change in the quadrupole
   alone with a single amplitude and no rescaling gives 96 ± 1 per cent, and
   including the rescaling 100 ± 2 per cent.

In real space the model quadrupole is zero; the real-space log-normal mock
`ln1real*` in `lrg_injected.npz` tests the construction there.

## Notes on the CIP fits

B_CIP enters only through B_X,Δ² B_CIP, so the posterior has a funnel shape.
We sample it with dynamic nested sampling and with the reparametrisation
v = B_CIP (B_n,Δ² + B_L,Δ²). Static nested sampling with random-walk proposals
stays at small B_CIP and misses the wide part of the funnel, where B_CIP is
large and B_X,Δ small (Appendix E). The marginal B_CIP depends on
the prior and the sampler, so only the product B_L,Δ² B_CIP is quoted. The
prior on B_CIP excludes negative values.

## Provenance

Software. Python 3.10.20 with numpy 2.2.6 (OpenBLAS 0.3.33), scipy 1.15.2,
dynesty 3.1.0, matplotlib 3.9.4 and corner 2.3.0 for the scripts here; the
measurements used pycorr 1.0.0 (git commit eaac8808), Corrfunc 2.5.1 (the
adematti fork, commit 1dbddfca), astropy 6.1.7, pandas 2.3.3, scikit-learn
1.7.2, healpy 1.20.0 and CAMB 1.6.6, and the halo catalogues were read with
abacusutils 2.1.2.

Galaxy catalogues. DESI DR1 large-scale-structure catalogues
(`dr1/survey/catalogs/dr1/LSS/iron/LSScats/`): v1.5pip for the LRGs, whose
completeness weights are individual inverse probabilities of fibre assignment,
with the first random file of each cap; v1.5 for BGS_BRIGHT-21.5, with four
random files per cap subsampled to 50 times the data. The faint BGS sample
(M_r < −20.5, 0.02 < z < 0.195) uses two BGS_BRIGHT random files per cap,
subsampled to ten times the data, with redshifts drawn from the weighted data
n(z). Galaxies are weighted by WEIGHT × WEIGHT_FKP and placed with a flat
ΛCDM distance–redshift relation with Ω_m = 0.3153, in Mpc/h. The LRG sample
has 2,138,575 galaxies after the photometric match and 14,349,309 randoms.

Photometry. Legacy Surveys DR9 photometry from the NOIRLab Astro Data Lab,
tables `desi_dr1.zpix` and `desi_dr1.photometry`, retrieved on 2026-07-24
(fluxes) and 2026-07-25 and 2026-07-27 (imaging maps) with the queries under
Data not included. Marks use the extinction-corrected fluxes,
flux / mw_transmission. Stellar masses are from FastSpecFit (iron, v2.1).

Mocks and simulations. DESI DR1 EZmocks
(`dr1/survey/catalogs/dr1/mocks/EZmock/dark/v1`, LRG_ffa, mocks 1–1000), cut
to 0.4 < z < 1.1, with the four random files of mock 1 shared by all mocks.
The mocks carry marks drawn from the measured W1 mark distribution and
modulated by 1 + 0.0205 δ_R (δ_R smoothed on 20 Mpc/h). The halo-occupation
test uses the z = 0.8 halo catalogues of AbacusSummit_base_c000_ph000–ph004.

Templates. CAMB 1.6.6 linear power spectra with H0 = 67.36, Ω_b h² = 0.02237,
Ω_c h² = 0.1200, n_s = 0.9649, A_s = 2.083 × 10⁻⁹ and one massive neutrino of
0.06 eV; f = Ω_m(z)^0.55; effective redshifts 0.7577 (LRG), 0.2891 (BGS) and
0.1583 (faint BGS).

Seeds. Nested sampling `default_rng(3)`; jackknife regions from pycorr's
`KMeansSubsampler` (angular, nside 512, `random_state=42`) on the data
randoms, and on the mock randoms for the mock jackknifes; mock marks
`default_rng(1000 + n)` for mock n; half splits of the calibration
`default_rng(1)`; mark shuffles `default_rng(123)`; BGS random subsampling
`default_rng(20260726)`; halo occupation `1000 + n` for box n.

## Data not included

The DESI DR1 catalogues are at https://data.desi.lbl.gov; v1.5 is also
mirrored at https://desidata.s3.amazonaws.com with the same paths, but v1.5pip
and the EZmocks are not. The photometry for the marks and the imaging maps
can be retrieved from the NOIRLab Astro Data Lab
(https://datalab.noirlab.edu/tap):

```sql
SELECT z.targetid, z.z, z.zwarn, p.flux_g, p.flux_r, p.flux_z, p.flux_w1,
       p.flux_ivar_r, p.flux_ivar_z, p.flux_ivar_w1, p.mw_transmission_g,
       p.mw_transmission_r, p.mw_transmission_z, p.mw_transmission_w1, p.ebv
FROM desi_dr1.zpix z JOIN desi_dr1.photometry p ON z.targetid=p.targetid
WHERE z.survey='main' AND z.program='dark' AND z.main_primary='true'
  AND MOD(z.desi_target,2)=1

SELECT z.targetid, p.psfdepth_g, p.psfdepth_r, p.psfdepth_z, p.galdepth_z,
       p.ebv, p.nobs_z, p.flux_ivar_w1, p.wisemask_w1, p.elat
FROM desi_dr1.zpix z JOIN desi_dr1.photometry p ON z.targetid=p.targetid
WHERE z.survey='main' AND z.program='dark' AND z.main_primary='true'
  AND MOD(z.desi_target,2)=1
```

`flux_ivar_w1` is not a depth map: in forced photometry it is dominated by
the source's own Poisson noise and so correlates with the W1 flux.

The AbacusSummit halo catalogues are at
https://data.desi.lbl.gov/public/cosmosim/AbacusSummit.
