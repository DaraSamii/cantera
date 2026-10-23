"""Cantera utilities for the ASEC reactive-flows onboarding project.

Subpackages
-----------
general
    Equilibrium calculations, parameter sweeps and result saving.
plots
    Plot style (LaTeX fonts, colourblind-safe palette), plotting helpers and
    figure saving.

Outputs are grouped by project part. Call :func:`set_part` once per notebook;
tables then go to ``results/<part>/raw`` and figures to
``results/<part>/figures/<format>``.
"""

from pathlib import Path

#: Project root (the folder that contains ``src/`` and ``results/``).
ROOT = Path(__file__).resolve().parents[2]

#: Top-level results folder.
RESULTS_DIR = ROOT / "results"

_part = None


# %============================================================================
def set_part(name):
    """
    Choose the project part that results and figures are saved under.

    Parameters
    ----------
    name : str
        Part name, e.g. ``"part1"``.

    Returns
    -------
    pathlib.Path
        The part's results folder, ``results/<name>``.
    """
    global _part
    _part = name
    return RESULTS_DIR / name


# %============================================================================
def part_dir():
    """
    Results folder of the current part.

    Returns
    -------
    pathlib.Path
        ``results/<part>`` if :func:`set_part` was called, else ``results``.
    """
    return RESULTS_DIR / _part if _part else RESULTS_DIR


# %============================================================================
def raw_dir():
    """
    Folder for raw tabular results of the current part.

    Returns
    -------
    pathlib.Path
        ``results/<part>/raw``.
    """
    return part_dir() / "raw"


# %============================================================================
def figures_dir():
    """
    Folder for figures of the current part (one subfolder per format).

    Returns
    -------
    pathlib.Path
        ``results/<part>/figures``.
    """
    return part_dir() / "figures"
