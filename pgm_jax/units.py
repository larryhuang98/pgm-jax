"""Units and physical constants: the single place where pgm_jax defines them.

Internal units (all library arguments, attributes, states and results): length nm, time ps, mass
amu (g/mol), energy kJ/mol, temperature K, pressure bar, charge e, dipole e nm, polarizability
nm^3, electric field V/nm, angles rad.  A value in any other unit carries the unit in its name
(``xyz_A``, ``dt_fs``, ``energy_kcal``; docs/api_design.md, 3.3).

Where the code base used slightly different values of one constant (different CODATA releases),
each value is kept under its own name, so that no result changes; unifying them would change
numbers and needs a decision of its own.  The three Bohr radii (BOHR_NM = CODATA 2014,
BOHR_NM_CODATA2018, BOHR_NM_CODATA2022) stay separate until the owner decides which one to keep
(docs/api_design.md, 9.1).
"""

# ----------------------------------------------------------------------------- thermodynamics
KB = 0.0083144626181532  # Boltzmann constant, kJ/mol/K (R / 1000, CODATA 2018)
BAR_PER_KJMOL_NM3 = 16.605390671738466  # 1 kJ/mol/nm^3 in bar
KJMOL_NM3_PER_BAR = 1.0 / BAR_PER_KJMOL_NM3  # 1 bar in kJ/mol/nm^3
AMU_NM3_TO_G_CM3 = 1.66053906660e-3  # a density of 1 amu/nm^3 in g/cm^3
HBAR_KJMOL_PS = 0.0635077993  # reduced Planck constant, kJ/mol ps (1.054571817e-34 J s x N_A)

# ----------------------------------------------------------------------------- energy, length, charge
KCAL = 4.184  # kJ/mol per kcal/mol (thermochemical calorie)
ANG_NM = 0.1  # nm per Angstrom
KJMOL_TO_MEV = 10.364269656262175  # 1 kJ/mol per particle in meV
HARTREE_KJMOL = 2625.4996394799  # kJ/mol per Hartree (CODATA 2018)
BOHR_NM = 0.052917721067  # nm per Bohr (CODATA 2014): pGM-pol table, PyRESP files, bonded-study data
BOHR_NM_CODATA2018 = 0.0529177210903  # nm per Bohr (CODATA 2018): residue-library import
BOHR_NM_CODATA2022 = 0.0529177210544  # nm per Bohr (CODATA 2022): the i-PI convention
DEBYE_E_NM = 0.020819434  # 1 D in e nm
KE = 138.935458  # Coulomb constant 1 / (4 pi eps0), kJ mol^-1 nm e^-2 (CODATA)

# The pGM code of Amber (sander pGM_multipoles.F90, pmemd-pgm) uses the Coulomb constant of Tinker,
# 332.05382 kcal A/mol, which is 2.98e-5 lower than CODATA (332.06371).  We keep CODATA as the
# model's constant; parity tests against Amber rescale by KE_AMBER_PGM / KE.
KE_AMBER_PGM = 332.05382 * 4.184 / 10.0  # kJ mol^-1 nm e^-2

# ----------------------------------------------------------------------------- SI and fields
E_CHARGE_C = 1.602176634e-19  # elementary charge, C
E_NM_C_M = E_CHARGE_C * 1e-9  # 1 e nm in C m
EPS0_SI = 8.8541878128e-12  # vacuum permittivity, F/m
KB_SI = 1.380649e-23  # Boltzmann constant, J/K
C_LIGHT_M_S = 2.99792458e8  # speed of light, m/s
C_CM_PS = 0.0299792458  # speed of light, cm/ps
FARADAY_KJ = 96.48533212331002  # kJ/mol per (e V): e N_A
