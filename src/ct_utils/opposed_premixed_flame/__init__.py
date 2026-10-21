"""Counterflow (opposed-jet) diffusion flames and their mixture-fraction analysis.

Fuel enters at x = 0 and oxidiser at x = width. The mixture fraction Z is the
Bilger mixture fraction: Z = 1 in the fuel stream and Z = 0 in the oxidiser.
"""

import cantera as ct
import numpy as np
import pandas as pd
from scipy.special import erfinv

#: Air as used for the oxidiser stream (mole fractions).
AIR_STREAM = "O2:0.21, N2:0.79"


def solve_counterflow(strain_rate, mechanism="h2o2.yaml", fuel="H2",
                      T_fuel=300.0, T_ox=300.0, p=101325.0, width=0.02):
    """
    Solve a fuel/air counterflow diffusion flame at a given strain rate.

    Both inlets get the same mass flux, set from the target strain rate as
    ``mdot = a * (width / 2) * (rho_f + rho_ox) / 2`` (project formula).

    Parameters
    ----------
    strain_rate : float
        Target strain rate in 1/s.
    mechanism : str, optional
        Cantera mechanism file. Default ``"h2o2.yaml"``.
    fuel : str, optional
        Fuel species name; the fuel stream is pure fuel. Default ``"H2"``.
    T_fuel : float, optional
        Fuel stream temperature in K. Default 300 K.
    T_ox : float, optional
        Oxidiser (air) stream temperature in K. Default 300 K.
    p : float, optional
        Pressure in Pa. Default 1 atm.
    width : float, optional
        Distance between the nozzles in m. Default 0.02 m.

    Returns
    -------
    cantera.CounterflowDiffusionFlame
        Solved flame object.
    """
    gas = ct.Solution(mechanism)
    fuel_stream = f"{fuel}:1"

    gas.TPX = T_fuel, p, fuel_stream
    rho_f = gas.density
    gas.TPX = T_ox, p, AIR_STREAM
    rho_ox = gas.density
    mdot = strain_rate * 0.5 * width * 0.5 * (rho_f + rho_ox)

    flame = ct.CounterflowDiffusionFlame(gas, width=width)
    flame.P = p
    flame.fuel_inlet.X, flame.fuel_inlet.T = fuel_stream, T_fuel
    flame.oxidizer_inlet.X, flame.oxidizer_inlet.T = AIR_STREAM, T_ox
    flame.fuel_inlet.mdot = mdot
    flame.oxidizer_inlet.mdot = mdot

    flame.set_refine_criteria(ratio=3, slope=0.05, curve=0.1)
    flame.solve(loglevel=0, auto=True)
    return flame


def stoichiometric_mixture_fraction(fuel="H2", oxidizer="O2:1, N2:3.76",
                                    mechanism="h2o2.yaml"):
    """
    Mixture fraction of the stoichiometric mixture, Z_st.

    Uses a separate gas object, so no flame state is modified.

    Parameters
    ----------
    fuel : str, optional
        Fuel composition string. Default ``"H2"``.
    oxidizer : str, optional
        Oxidiser composition string. Default air.
    mechanism : str, optional
        Cantera mechanism file. Default ``"h2o2.yaml"``.

    Returns
    -------
    float
        Z_st (dimensionless); about 0.0285 for H2/air.
    """
    gas = ct.Solution(mechanism)
    gas.set_equivalence_ratio(1.0, fuel, oxidizer)
    return gas.mixture_fraction(fuel, oxidizer)


def scalar_dissipation_rate(flame):
    """
    Scalar dissipation rate chi(x) = 2 D |dZ/dx|^2 across a solved flame.

    Cantera has no built-in chi, so it is computed from the Bilger mixture
    fraction with a unity-Lewis-number diffusivity D = k / (rho cp).

    Parameters
    ----------
    flame : cantera.CounterflowDiffusionFlame
        Solved flame.

    Returns
    -------
    numpy.ndarray
        chi in 1/s at every grid point.
    """
    Z = flame.mixture_fraction("Bilger")
    D = flame.thermal_conductivity / (flame.density * flame.cp_mass)
    return 2 * D * np.gradient(Z, flame.grid) ** 2


def extract_TZ_profile(flame, fuel="H2", oxidizer="O2:1, N2:3.76"):
    """
    Extract temperature and mixture-fraction profiles from a solved flame.

    Parameters
    ----------
    flame : cantera.CounterflowDiffusionFlame
        Solved flame.
    fuel : str, optional
        Fuel composition string, used for Z_st. Default ``"H2"``.
    oxidizer : str, optional
        Oxidiser composition string, used for Z_st. Default air.

    Returns
    -------
    Z : numpy.ndarray
        Mixture fraction at each grid point.
    T : numpy.ndarray
        Temperature in K at each grid point.
    chi_st : float
        Scalar dissipation rate (1/s) at the grid point closest to Z_st.
    """
    Z = flame.mixture_fraction("Bilger")
    Z_st = stoichiometric_mixture_fraction(fuel, oxidizer, flame.gas.source)
    chi = scalar_dissipation_rate(flame)
    return Z, flame.T, chi[np.argmin(np.abs(Z - Z_st))]


def chi_st_analytic(strain_rate, Z_st):
    """
    Analytical chi at Z_st for a constant-density counterflow.

    chi_st = (a / pi) exp(-2 [erfinv(1 - 2 Z_st)]^2)

    Parameters
    ----------
    strain_rate : float or array-like
        Strain rate a in 1/s.
    Z_st : float
        Stoichiometric mixture fraction.

    Returns
    -------
    float or numpy.ndarray
        chi_st in 1/s.
    """
    return np.asarray(strain_rate) / np.pi * np.exp(-2 * erfinv(1 - 2 * Z_st) ** 2)


def equilibrium_T_of_Z(Z_range, fuel="H2", T_fuel=300.0, T_ox=300.0,
                       p=101325.0, mechanism="h2o2.yaml", oxidizer=AIR_STREAM):
    """
    Equilibrium temperature as a function of mixture fraction.

    At each Z the two streams are blended by mass, Y = Z Y_f + (1 - Z) Y_ox,
    with the enthalpy blended the same way, h = Z h_f + (1 - Z) h_ox, then
    equilibrated at constant h and p. The result is the upper temperature
    bound for any adiabatic non-premixed flame of these streams.

    Parameters
    ----------
    Z_range : array-like
        Mixture fractions in [0, 1].
    fuel : str, optional
        Fuel species name. Default ``"H2"``.
    T_fuel, T_ox : float, optional
        Stream temperatures in K. Default 300 K.
    p : float, optional
        Pressure in Pa. Default 1 atm.
    mechanism : str, optional
        Cantera mechanism file. Default ``"h2o2.yaml"``.
    oxidizer : str, optional
        Oxidiser composition string. Default ``"O2:0.21, N2:0.79"``.

    Returns
    -------
    numpy.ndarray
        T_eq in K at each Z.
    """
    Z_range = np.asarray(Z_range, dtype=float)
    gas = ct.Solution(mechanism)

    gas.TPX = T_fuel, p, f"{fuel}:1"
    Y_f, h_f = gas.Y.copy(), gas.enthalpy_mass
    gas.TPX = T_ox, p, oxidizer
    Y_ox, h_ox = gas.Y.copy(), gas.enthalpy_mass

    T_eq = np.empty_like(Z_range)
    for i, Z in enumerate(Z_range):
        gas.HPY = Z * h_f + (1 - Z) * h_ox, p, Z * Y_f + (1 - Z) * Y_ox
        gas.equilibrate("HP")
        T_eq[i] = gas.T
    return T_eq


def flame_to_dataframe(flame):
    """
    Export a solved flame's profiles with ``to_pandas``, plus Z and chi.

    Parameters
    ----------
    flame : cantera.CounterflowDiffusionFlame
        Solved flame.

    Returns
    -------
    pandas.DataFrame
        One row per grid point: ``x`` (m), ``u`` (m/s), ``Z``, ``chi`` (1/s),
        then ``T`` (K), ``P`` (Pa) and every ``X_<species>``.
    """
    df = flame.to_array().to_pandas(cols=("T", "P", "X"))
    df.insert(0, "chi", scalar_dissipation_rate(flame))
    df.insert(0, "Z", flame.mixture_fraction("Bilger"))
    df.insert(0, "u", flame.velocity)
    df.insert(0, "x", flame.grid)
    return df
