# pGM-JAX paper

`main.tex` + `references.bib` -> `main.pdf` (`latexmk -pdf main.tex`). Authors and affiliations
are placeholders.

| Path | Content |
|---|---|
| `figures/make_figures.py` | all data figures from `data/` (`python paper/figures/make_figures.py`) |
| `data/speed.json` | MD speed (README benchmark, `scripts/benchmarks/bench_md.py`) |
| `data/water_ref.json`, `data/water_recovery.json` | water LJ parameter-recovery fit (`scripts/fitting/fit_liquid.py water ...`, `runs/liquid/water_recovery.sh`) |
| `data/methanol.json` | methanol LJ fit to experiment (`scripts/fitting/fit_liquid.py methanol --iters 6`) |
| `data/flex_methanol.json` | flexible methanol checks (`paper/scripts/flex_methanol.py`) |

Reproduce the runs (one GPU each; wall times on an RTX PRO 6000 Blackwell):

```bash
python examples/flex_methanol_check.py                 # fits runs/flex/methanol.flex (bonded, ~min)
python paper/scripts/flex_methanol.py                  # ~7 min
python scripts/fitting/fit_liquid.py water --iters 1 --start 0,0 --equil0-ps 100 --prod-ps 400 --out runs/liquid/water_ref.json
python scripts/fitting/fit_liquid.py water --start 0.02956,-0.35667 --iters 8 --targets 1.01768,8.6377 --out runs/liquid/water_recovery.json   # ~30 min
python scripts/fitting/fit_liquid.py methanol --iters 6 --out runs/liquid/methanol.json     # ~1.5 h
cp runs/liquid/{water_ref,water_recovery,methanol}.json paper/data/
```
