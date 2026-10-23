"""Post-processing helpers: species extraction, curve crossings, fits, thermal-NO rate."""

import numpy as np


# %============================================================================
def mole_fractions(states, species):
    """
    Extract the mole fractions of selected species from a SolutionArray.

    Parameters
    ----------
    states : cantera.SolutionArray
        States to read from.
    species : list of str
        Species names, e.g. ``["OH", "H", "O"]``.

    Returns
    -------
    dict of str to numpy.ndarray
        Mole fraction array for each species, keyed by name.
    """
    names = states.species_names
    return {sp: states.X[..., names.index(sp)] for sp in species}


# %============================================================================
def crossings(x, y, level):
    """
    Find where a sampled curve y(x) crosses a given level.

    Uses linear interpolation between the two samples on either side of
    each crossing.

    Parameters
    ----------
    x : array-like
        Sample positions, monotonic.
    y : array-like
        Sampled values, same length as ``x``.
    level : float
        Level to look for.

    Returns
    -------
    numpy.ndarray
        x positions of every crossing, in increasing order of index.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    d = y - level
    idx = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    return x[idx] - d[idx] * (x[idx + 1] - x[idx]) / (d[idx + 1] - d[idx])


# %============================================================================
def linear_slope(x, y):
    """
    Least-squares slope dy/dx of a straight-line fit.

    Parameters
    ----------
    x, y : array-like
        Data points.

    Returns
    -------
    float
        Slope of the fitted line, in units of y per unit of x.
    """
    return np.polyfit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), 1)[0]


# %============================================================================
def power_law_fit(x, y):
    """
    Fit a power law y = C x^n (a straight line on log-log axes).

    Parameters
    ----------
    x, y : array-like
        Positive data points.

    Returns
    -------
    C : float
        Prefactor, in units of y / x^n.
    n : float
        Exponent (dimensionless).
    """
    n, ln_C = np.polyfit(np.log(x), np.log(y), 1)
    return np.exp(ln_C), n


# %============================================================================
def power_law_exponent(x, y):
    """
    Exponent n of a power-law fit y = C x^n (slope on log-log axes).

    Parameters
    ----------
    x, y : array-like
        Positive data points.

    Returns
    -------
    float
        Fitted exponent n (dimensionless).
    """
    return power_law_fit(x, y)[1]


# %============================================================================
def zeldovich_no_rate(states):
    """
    Initial thermal-NO formation rate from the Zeldovich rate-limiting step.

    Uses d[NO]/dt = 2 k1 [O][N2] with k1 = 1.8e11 exp(-38370/T) m^3/(kmol s)
    for O + N2 -> NO + N, evaluated at each (equilibrium) state, and converts
    it to a mole-fraction rate. The mechanism itself needs no N chemistry,
    only the O and N2 species.

    Parameters
    ----------
    states : cantera.SolutionArray or cantera.Solution
        States containing the species O and N2.

    Returns
    -------
    numpy.ndarray or float
        NO formation rate in ppm per millisecond.
    """
    names = states.species_names
    c = states.concentrations                          # kmol/m^3
    k1 = 1.8e11 * np.exp(-38370.0 / states.T)          # m^3/(kmol s)
    rate = 2 * k1 * c[..., names.index("O")] * c[..., names.index("N2")]
    return rate / states.density_mole * 1e6 / 1e3      # 1/s -> ppm/ms
