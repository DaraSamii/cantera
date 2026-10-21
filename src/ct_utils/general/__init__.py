"""Equilibrium calculations, parameter sweeps and result saving.

All equilibrium calculations here are adiabatic and isobaric: the reactants are
set to ``(T_u, p)`` at a given equivalence ratio and then equilibrated at
constant enthalpy and pressure (``"HP"``). The final temperature is the
adiabatic flame temperature ``T_ad``.

Every array-based function goes through :func:`equilibrium_states`, which
broadcasts ``phi``, ``T_u`` and ``p`` against each other. The sweep and map
functions are thin wrappers around it, so no sweep logic is duplicated.
"""

from pathlib import Path

import cantera as ct
import numpy as np
import pandas as pd

from ct_utils import raw_dir

#: Air as 1 mol O2 + 3.76 mol N2.
AIR = "O2:1, N2:3.76"


# ── Single states ─────────────────────────────────────────────────────────────

def equilibrium_state(phi, fuel="H2", T_u=300.0, p=101325.0,
                      mechanism="h2o2.yaml", oxidizer=AIR):
    """
    Return the HP-equilibrium state of a premixed fuel/oxidiser mixture.

    Parameters
    ----------
    phi : float
        Equivalence ratio.
    fuel : str, optional
        Cantera fuel composition string. Default ``"H2"``.
    T_u : float, optional
        Unburned (reactant) temperature in K. Default 300 K.
    p : float, optional
        Pressure in Pa. Default 1 atm.
    mechanism : str, optional
        Cantera mechanism file. Default ``"h2o2.yaml"``.
    oxidizer : str, optional
        Oxidiser composition string. Default air, ``"O2:1, N2:3.76"``.

    Returns
    -------
    cantera.Solution
        Gas object at the equilibrium state, so any property (T, X, Y, ...)
        can be read from it.
    """
    gas = ct.Solution(mechanism)
    gas.set_equivalence_ratio(phi, fuel, oxidizer)
    gas.TP = T_u, p
    gas.equilibrate("HP")
    return gas


def equilibrium_temperature(phi, fuel="H2", T_u=300.0, p=101325.0,
                            mechanism="h2o2.yaml", oxidizer=AIR):
    """
    Return the adiabatic flame temperature for a single condition.

    Parameters
    ----------
    phi : float
        Equivalence ratio.
    fuel : str, optional
        Cantera fuel composition string. Default ``"H2"``.
    T_u : float, optional
        Unburned temperature in K. Default 300 K.
    p : float, optional
        Pressure in Pa. Default 1 atm.
    mechanism : str, optional
        Cantera mechanism file. Default ``"h2o2.yaml"``.
    oxidizer : str, optional
        Oxidiser composition string. Default air.

    Returns
    -------
    float
        Adiabatic flame temperature in K.
    """
    return equilibrium_state(phi, fuel, T_u, p, mechanism, oxidizer).T


# ── Arrays of states ──────────────────────────────────────────────────────────

def equilibrium_states(phi, T_u=300.0, p=101325.0, fuel="H2",
                       mechanism="h2o2.yaml", oxidizer=AIR):
    """
    Equilibrate many conditions at once and return them as a SolutionArray.

    ``phi``, ``T_u`` and ``p`` are broadcast against each other with NumPy
    rules, then flattened to 1D. The result can be exported directly with
    ``states.to_pandas()``.

    Parameters
    ----------
    phi : float or array-like
        Equivalence ratio(s).
    T_u : float or array-like, optional
        Unburned temperature(s) in K. Default 300 K.
    p : float or array-like, optional
        Pressure(s) in Pa. Default 1 atm.
    fuel : str, optional
        Cantera fuel composition string. Default ``"H2"``.
    mechanism : str, optional
        Cantera mechanism file. Default ``"h2o2.yaml"``.
    oxidizer : str, optional
        Oxidiser composition string. Default air.

    Returns
    -------
    cantera.SolutionArray
        1D array of equilibrium states, with extra columns ``phi`` and ``T_u``
        holding the inputs of each state.
    """
    phi, T_u, p = (np.ravel(a) for a in
                   np.broadcast_arrays(np.asarray(phi, dtype=float),
                                       np.asarray(T_u, dtype=float),
                                       np.asarray(p, dtype=float)))
    states = ct.SolutionArray(ct.Solution(mechanism), shape=phi.shape,
                              extra={"phi": phi, "T_u": T_u})
    states.set_equivalence_ratio(phi, fuel, oxidizer)
    states.TP = T_u, p
    states.equilibrate("HP")
    return states


def equilibrium_sweep(phi_range, fuel="H2", T_u=300.0, p=101325.0,
                      mechanism="h2o2.yaml", oxidizer=AIR):
    """
    Return T_ad for an array of equivalence ratios.

    Parameters
    ----------
    phi_range : array-like
        Equivalence ratios.
    fuel : str, optional
        Cantera fuel composition string. Default ``"H2"``.
    T_u : float, optional
        Unburned temperature in K. Default 300 K.
    p : float, optional
        Pressure in Pa. Default 1 atm.
    mechanism : str, optional
        Cantera mechanism file. Default ``"h2o2.yaml"``.
    oxidizer : str, optional
        Oxidiser composition string. Default air.

    Returns
    -------
    numpy.ndarray
        Adiabatic flame temperatures in K, same shape as ``phi_range``.
    """
    phi_range = np.asarray(phi_range, dtype=float)
    states = equilibrium_states(phi_range, T_u, p, fuel, mechanism, oxidizer)
    return states.T.reshape(phi_range.shape)


def equilibrium_sweep_Tu(T_u_range, phi=1.0, fuel="H2", p=101325.0,
                         mechanism="h2o2.yaml", oxidizer=AIR):
    """
    Return T_ad for an array of unburned temperatures.

    Parameters
    ----------
    T_u_range : array-like
        Unburned temperatures in K.
    phi : float, optional
        Equivalence ratio. Default 1.0.
    fuel : str, optional
        Cantera fuel composition string. Default ``"H2"``.
    p : float, optional
        Pressure in Pa. Default 1 atm.
    mechanism : str, optional
        Cantera mechanism file. Default ``"h2o2.yaml"``.
    oxidizer : str, optional
        Oxidiser composition string. Default air.

    Returns
    -------
    numpy.ndarray
        Adiabatic flame temperatures in K, same shape as ``T_u_range``.
    """
    T_u_range = np.asarray(T_u_range, dtype=float)
    states = equilibrium_states(phi, T_u_range, p, fuel, mechanism, oxidizer)
    return states.T.reshape(T_u_range.shape)


def equilibrium_sweep_p(p_range, phi=1.0, fuel="H2", T_u=300.0,
                        mechanism="h2o2.yaml", oxidizer=AIR):
    """
    Return T_ad for an array of pressures.

    Parameters
    ----------
    p_range : array-like
        Pressures in Pa.
    phi : float, optional
        Equivalence ratio. Default 1.0.
    fuel : str, optional
        Cantera fuel composition string. Default ``"H2"``.
    T_u : float, optional
        Unburned temperature in K. Default 300 K.
    mechanism : str, optional
        Cantera mechanism file. Default ``"h2o2.yaml"``.
    oxidizer : str, optional
        Oxidiser composition string. Default air.

    Returns
    -------
    numpy.ndarray
        Adiabatic flame temperatures in K, same shape as ``p_range``.
    """
    p_range = np.asarray(p_range, dtype=float)
    states = equilibrium_states(phi, T_u, p_range, fuel, mechanism, oxidizer)
    return states.T.reshape(p_range.shape)


def equilibrium_map(phi_range, sweep_range, sweep_param="T_u",
                    fuel="H2", fixed_T_u=300.0, fixed_p=101325.0,
                    mechanism="h2o2.yaml", oxidizer=AIR, return_states=False):
    """
    Compute a 2D map of T_ad over (phi, sweep_param).

    Parameters
    ----------
    phi_range : array-like
        Equivalence ratio axis (rows of the map).
    sweep_range : array-like
        Values of the swept parameter (columns of the map): T_u in K or p in Pa.
    sweep_param : {"T_u", "p"}, optional
        Which parameter to sweep on the second axis. Default ``"T_u"``.
    fuel : str, optional
        Cantera fuel composition string. Default ``"H2"``.
    fixed_T_u : float, optional
        Unburned temperature in K, used when ``sweep_param="p"``. Default 300 K.
    fixed_p : float, optional
        Pressure in Pa, used when ``sweep_param="T_u"``. Default 1 atm.
    mechanism : str, optional
        Cantera mechanism file. Default ``"h2o2.yaml"``.
    oxidizer : str, optional
        Oxidiser composition string. Default air.
    return_states : bool, optional
        If True, also return the flattened SolutionArray. Default False.

    Returns
    -------
    numpy.ndarray, shape (len(phi_range), len(sweep_range))
        T_ad in K at each (phi, sweep_param) point.
    cantera.SolutionArray
        Only if ``return_states`` is True: all equilibrium states, flattened
        row by row.

    Raises
    ------
    ValueError
        If ``sweep_param`` is not ``"T_u"`` or ``"p"``.
    """
    if sweep_param not in ("T_u", "p"):
        raise ValueError(f"sweep_param must be 'T_u' or 'p', got {sweep_param!r}")

    PHI, SWEEP = np.meshgrid(np.asarray(phi_range, dtype=float),
                             np.asarray(sweep_range, dtype=float), indexing="ij")
    T_u = SWEEP if sweep_param == "T_u" else fixed_T_u
    p = SWEEP if sweep_param == "p" else fixed_p

    states = equilibrium_states(PHI, T_u, p, fuel, mechanism, oxidizer)
    T_map = states.T.reshape(PHI.shape)
    return (T_map, states) if return_states else T_map


# ── Post-processing ───────────────────────────────────────────────────────────

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
    return linear_slope(np.log(x), np.log(y))


def preheat_slope(phi_range, T_u_range, p=101325.0, **kwargs):
    """
    Slope dT_ad/dT_u for each equivalence ratio.

    For each phi, T_ad is computed over ``T_u_range`` and a straight line is
    fitted to T_ad(T_u).

    Parameters
    ----------
    phi_range : array-like
        Equivalence ratios.
    T_u_range : array-like
        Unburned temperatures in K used for the fit.
    p : float, optional
        Pressure in Pa. Default 1 atm.
    **kwargs
        Passed to :func:`equilibrium_map` (fuel, mechanism, oxidizer).

    Returns
    -------
    numpy.ndarray
        dT_ad/dT_u (K/K) for each phi, same length as ``phi_range``.
    """
    T_map = equilibrium_map(phi_range, T_u_range, "T_u", fixed_p=p, **kwargs)
    return np.array([linear_slope(T_u_range, row) for row in T_map])


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


# ── Saving ────────────────────────────────────────────────────────────────────

def save_dataframe(df, name, folder=None):
    """
    Save a DataFrame as CSV in the current part's ``raw`` folder.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to save.
    name : str
        File name without extension.
    folder : path-like, optional
        Output folder. Default ``results/<part>/raw`` (see ``ct_utils.set_part``).

    Returns
    -------
    pathlib.Path
        Path of the written file.
    """
    folder = Path(folder) if folder is not None else raw_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def save_states(states, name, folder=None):
    """
    Export a SolutionArray with ``to_pandas`` and save it as CSV.

    The columns are the inputs ``phi`` and ``T_u``, then ``T`` (K), ``P`` (Pa)
    and the mole fraction of every species (``X_<name>``).

    Parameters
    ----------
    states : cantera.SolutionArray
        1D array of states, as returned by :func:`equilibrium_states`.
    name : str
        File name without extension.
    folder : path-like, optional
        Output folder. Default ``results/<part>/raw``.

    Returns
    -------
    pandas.DataFrame
        The exported table (also written to disk).
    """
    df = states.to_pandas(cols=("phi", "T_u", "T", "P", "X"))
    save_dataframe(df, name, folder)
    return df
