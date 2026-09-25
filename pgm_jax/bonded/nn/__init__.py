"""NNB: fast neural bonded terms for pGM molecules ("nn" term set).

The network replaces *atom typing*, not the energy function: the bonded energy keeps the form of
a chosen set of physical term families (default: the class II set, terms.PAPER; any families of
terms.REGISTRY work), and a graph network predicts the parameters of every term instance.  At MD
time the parameters are fixed tables, so the speed is that of the classical family set.

  Stage 1, topology -> parameters (once per molecule or residue template).  Message passing over
  the bond graph gives atom embeddings h_i from the element, degree, bond orders, ring and
  aromatic flags and the atom's pGM electrostatic parameters (charge, polarizability, Gaussian
  width, covalent-dipole strength): the bonded model knows the all-pair electrostatic model it
  complements.  Every term instance is then described exactly as a typed force field would key
  it (the families' own `index(top, keyf)` is called with a recording keyf), but with the atom
  classes replaced by symmetric readouts of the embeddings:
      atom a              h_a
      bond / pair (i, j)  [h_i + h_j, h_i * h_j]
      angle (i, j, k)     [h_j, h_i + h_k, h_i * h_k]
      torsion (i,j,k,l)   [h_j + h_k, h_j * h_k, h_i + h_l, h_i * h_l, h_i * h_j + h_l * h_k]
      improper (c;a,b,d)  [h_c, sum h_x, sum_{x<y} h_x * h_y]
  concatenated in the order of the key's components (so oriented couplings, e.g. bond-angle with
  a given outer atom, stay oriented), plus the key's literal skeleton.  One small MLP head per
  family maps this to the family's parameters (init + scale * output, output layers start at
  zero: the untrained model is the family set at its default values); two more heads give
  corrections to the bond and angle reference values.

  Stage 2, coordinates -> energy: the families' own energy functions with per-instance
  parameters.  `freeze` evaluates stage 1 once (FlexibleTemplate does it for MD).

Typed table + network residual (`table_depth`, `resid_l2`): with table_depth = d every head's
output is  table[typed key of the instance] + network residual,  the typed key being the one of
the classical terms with atom environments to depth d (0 = element-typed).  A penalty
resid_l2 * mean(residual^2) (training molecules) shrinks the residual, so the model is the typed
force field where the data do not ask for more, and an environment-aware one where they do.
The typed table is readable (bond C-H, angle H-C-O, ...) like a classical parameter file.

Sequence context (`context`, proteins): families listed in instances.CONTEXT_ATOMS (the backbone
correction map "cmap") also see the embeddings of the residues i-1, i, i+1 (mean of the atom
embeddings of each residue of Topology.residue, through one more MLP), so a residue's phi/psi map
depends on its neighbours; frozen for MD like everything else.

Modules: layers (MLPs, message passing), features (graph inputs), instances (key decomposition,
readouts), model (NNBConfig, Vocabulary, NNBonded: stage 1 / stage 2, save / load).

Reference values: `ref="geometry"` starts r0 and th0 from the molecule's minimum geometry (from
MACE-OFF or DFT, available for any new molecule); `ref="predicted"` from covalent radii and
hybridization angles.  Trained with pGM nonbonded in the loop (bonded/fit.py, Adam then L-BFGS).
"""

from .features import ELEMENTS, N_FEAT  # noqa: F401,E402
from .model import NNBConfig, NNBonded, Vocabulary  # noqa: F401,E402
