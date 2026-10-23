"""Freely propagating premixed flames: solver, flame properties and sweeps.

The unburned mixture enters at x = 0; the burned gas leaves at x = width.
"""

from functools import lru_cache

import cantera as ct
import numpy as np

from ct_utils.opposed_premixed_flame import stoichiometric_mixture_fraction


# %============================================================================
# shortcut: cache keeps every solved flame in memory (fine for ~100 H2 flames), clear with
# solve_free_flame.cache_clear() if a sweep gets large. It lets the S_L and delta_L sweeps share solves.
@lru_cache(maxsize=None)
def solve_free_flame(phi, T_u=300.0, p=101325.0,
                     transport="mixture-averaged",
                     fuel="H2", oxidizer="O2:1, N2:3.76",
                     mechanism="h2o2.yaml", width=0.03):
    """
    Solve a freely propagating premixed flame.

    Parameters
    ----------
    phi : float
        Equivalence ratio.
    T_u : float, optional
        Unburned mixture temperature in K. Default 300 K.
    p : float, optional
        Pressure in Pa. Default 1 atm.
    transport : {"mixture-averaged", "unity-Lewis-number"}, optional
        Transport model. Default ``"mixture-averaged"``.
    fuel : str, optional
        Fuel composition string. Default ``"H2"``.
    oxidizer : str, optional
        Oxidiser composition string. Default air.
    mechanism : str, optional
        Cantera mechanism file. Default ``"h2o2.yaml"``.
    width : float, optional
        Domain width in m. Default 0.03 m.

    Returns
    -------
    cantera.FreeFlame
        Solved flame object. Calls with identical arguments return the same
        (cached) object, so treat it as read-only.
    """
    gas = ct.Solution(mechanism, transport_model=transport)
    gas.set_equivalence_ratio(phi, fuel, oxidizer)
    gas.TP = T_u, p

    flame = ct.FreeFlame(gas, width=width)
    flame.set_refine_criteria(ratio=2.0, slope=0.005, curve=0.01)
    flame.set_max_grid_points(flame.flame, 10000)
    flame.solve(loglevel=0, auto=True)
    return flame


# %============================================================================
def flame_speed(flame):
    """
    Laminar flame speed S_L: the gas velocity at the unburned inlet.

    Parameters
    ----------
    flame : cantera.FreeFlame
        Solved flame.

    Returns
    -------
    float
        S_L in m/s.
    """
    return flame.velocity[0]


# %============================================================================
def flame_thickness(flame):
    """
    Thermal flame thickness delta_L = (T_b - T_u) / max(dT/dx).

    Parameters
    ----------
    flame : cantera.FreeFlame
        Solved flame.

    Returns
    -------
    float
        delta_L in m.
    """
    T, x = flame.T, flame.grid
    return float((T[-1] - T[0]) / np.max(np.gradient(T, x)))


# %============================================================================
def flame_position(flame):
    """
    Position where T = T_u + (T_b - T_u)/2, used to align flame fronts.

    Parameters
    ----------
    flame : cantera.FreeFlame
        Solved flame.

    Returns
    -------
    float
        Position in m (linear interpolation on the monotonic T profile).
    """
    T = flame.T
    return float(np.interp(T[0] + 0.5 * (T[-1] - T[0]), T, flame.grid))


# %============================================================================
def mixture_fraction(flame, fuel="H2", oxidizer="O2:1, N2:3.76"):
    """
    Mixture fraction Z at every grid point of a premixed flame.

    ``FreeFlame`` has no ``mixture_fraction`` method (only counterflow flames
    do), so Z is evaluated point by point with a separate gas object.

    Parameters
    ----------
    flame : cantera.FreeFlame
        Solved flame.
    fuel : str, optional
        Fuel composition string. Default ``"H2"``.
    oxidizer : str, optional
        Oxidiser composition string. Default air.

    Returns
    -------
    numpy.ndarray
        Z (dimensionless) at each grid point.
    """
    gas = ct.Solution(flame.gas.source)
    Z = np.empty(len(flame.grid))
    for i, Y in enumerate(flame.Y.T):          # flame.Y is (n_species, n_points)
        gas.Y = Y
        Z[i] = gas.mixture_fraction(fuel, oxidizer)
    return Z


# %============================================================================
def flame_speed_sweep(phi_range, T_u=300.0, p=101325.0,
                      transport="mixture-averaged",
                      fuel="H2", oxidizer="O2:1, N2:3.76",
                      mechanism="h2o2.yaml"):
    """
    Compute S_L over a range of equivalence ratios.

    Parameters
    ----------
    phi_range : array-like
        Equivalence ratios.
    T_u, p, transport, fuel, oxidizer, mechanism
        As in :func:`solve_free_flame`.

    Returns
    -------
    numpy.ndarray
        S_L in m/s, same shape as ``phi_range``.
    """
    phi_range = np.asarray(phi_range, dtype=float)
    S_L = np.empty_like(phi_range)

    for i, phi in np.ndenumerate(phi_range):
        flame = solve_free_flame(float(phi), T_u, p, transport, fuel, oxidizer, mechanism)
        S_L[i] = flame_speed(flame)

    return S_L


# %============================================================================
def flame_thickness_sweep(phi_range, T_u=300.0, p=101325.0,
                          transport="mixture-averaged",
                          fuel="H2", oxidizer="O2:1, N2:3.76",
                          mechanism="h2o2.yaml"):
    """
    Compute delta_L over a range of equivalence ratios.

    Parameters
    ----------
    phi_range : array-like
        Equivalence ratios.
    T_u, p, transport, fuel, oxidizer, mechanism
        As in :func:`solve_free_flame`.

    Returns
    -------
    numpy.ndarray
        delta_L in m, same shape as ``phi_range``.
    """
    phi_range = np.asarray(phi_range, dtype=float)
    delta_L = np.empty_like(phi_range)

    for i, phi in np.ndenumerate(phi_range):
        flame = solve_free_flame(float(phi), T_u, p, transport, fuel, oxidizer, mechanism)
        delta_L[i] = flame_thickness(flame)

    return delta_L


# %============================================================================
def flame_to_dataframe(flame):
    """
    Export a solved flame's profiles with ``to_pandas``.

    Parameters
    ----------
    flame : cantera.FreeFlame
        Solved flame.

    Returns
    -------
    pandas.DataFrame
        One row per grid point: ``x`` (m), ``u`` (m/s), ``hrr`` (W/m^3),
        then ``T`` (K), ``P`` (Pa) and every ``X_<species>``.
    """
    df = flame.to_array().to_pandas(cols=("T", "P", "X"))
    df.insert(0, "hrr", flame.heat_release_rate)
    df.insert(0, "u", flame.velocity)
    df.insert(0, "x", flame.grid)
    return df
