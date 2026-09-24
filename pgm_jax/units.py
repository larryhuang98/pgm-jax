"""Unit conventions used throughout pgm_jax.

Internal units: energy kJ/mol, length nm, charge e, angles rad.
"""

HARTREE_KJMOL = 2625.499639          # 1 Eh in kJ/mol
KCAL_KJMOL = 4.184
ANG_NM = 0.1
BOHR_NM = 0.052917721067
KE = 138.935458                       # Coulomb constant, kJ mol^-1 nm e^-2
DEBYE_E_NM = 0.020819434              # 1 D in e·nm

# Amber's pGM code (sander pGM_multipoles.F90, pmemd-pgm) uses Tinker's Coulomb constant,
# 332.05382 kcal A/mol, which is 2.98e-5 lower than CODATA (332.06371).  We keep CODATA as the
# model's constant; parity tests against Amber rescale by KE_AMBER_PGM / KE.
KE_AMBER_PGM = 332.05382 * 4.184 / 10.0     # kJ mol^-1 nm e^-2
