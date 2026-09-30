# Isotropic periodic sum (IPS)

`long_range="ips"` replaces PME (and the continuum vdW tail) by the isotropic periodic sum of Wu and
Brooks: every pair inside the cutoff interacts through the pair function plus a polynomial that makes
the energy and its first derivatives vanish at `rc`; pairs beyond `rc` are dropped and the periodic
images enter as self terms. No FFT, no grid, no Ewald splitting.

    MDSettings().replace(long_range="ips", ips_order=4, cutoff=0.9)
    python scripts/md/run_md.py ... --long-range ips --ips-order 4 [--ips-boundary]

## What is covered

- Electrostatics of the Gaussian charges and dipoles (`elec = q, qp, qi, qpi`): the pair function is
  erf(a r)/r + (1/rc) sum_{k<N} c_k (r/rc)^(2k), a = 1/sqrt(2(R_i^2 + R_k^2)); the c_k follow from
  Phi^(m)(rc) = 0, m < N, for each pair of widths (`ips.elec_coefficients`, differentiable in the radii).
  The kernels G_n = (-(1/r) d/dr)^n Phi feed the same row code as PME (permanent and induced dipoles, the
  field, the iel terms). Self images: charge (1/2) q^2 c_0 / rc, dipole -c_1 |d|^2 / rc^3.
- `ips_order` N = 2 .. 12, "same polynomial family, more terms". N = 4 is sander/pmemd's `aipseg`
  (agreement 1e-16); the point-charge limit is the AIPSE set (-35, 35, -21, 5) / 16.
- Lennard-Jones (`vdw="lj"`): sander's 3D-IPS (fixed AIPSVA / AIPSVC coefficients).
- Double exponential (`vdw="de"`): the analytic IPS of DEGAUSS (`ips_de.h`); the force vanishes at `rc`,
  the energy does not. `ips_boundary=True` adds pmemd's constant boundary energy (1/2 sum_ij Phi_ij(rc) f),
  f the fraction of pairs inside `rc`; it does not change forces.
- Not covered: GVDW, multiple time stepping (`MTS` raises), gas-phase / `PeriodicModel` IPS.

## Validation

512 pGM3P-25 waters (Gaussian charges, DE 18.17 / 3.65, 9 A), single point against pmemd DEGAUSS IPS:
electrostatics -5037.013 vs -5036.989 kcal/mol (5e-6 relative; sander -5036.989), vdW 624.8594 vs 624.8594
with `ips_boundary`, 662.257 without. NVE, 2 ps, 1 fs, double precision: drift 0.06 kJ/mol/ps on a total of
-2.1e6 kJ/mol (PME: 0.05). Electrostatic energy of the liquid agrees with PME to 3e-5 relative.
`tests/test_ips.py` holds the unit tests.
