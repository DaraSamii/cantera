# Cantera – Reactive Flows

Setup for the ASEC Lab onboarding chapter on reactive flows (0D → 1D → 3D).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/check_setup.py
```

## Tutorial (LaTeX)

`latex/main.pdf` is a from-scratch tutorial covering the whole onboarding chapter and project:
thermo, equilibrium, kinetics, 0D reactors, transport, reacting Navier–Stokes, premixed and
non-premixed flames, Cantera practice and a task-by-task project companion.

```bash
.venv/bin/python latex/figures/scripts/make_figures.py   # regenerate Cantera figures (~5 min)
cd latex && latexmk -pdf main.tex
```
