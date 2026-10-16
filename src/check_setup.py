"""Smoke test: 0D equilibrium and 1D free flame with GRI-Mech 3.0."""
import cantera as ct

gas = ct.Solution("gri30.yaml")
gas.TPX = 300, ct.one_atm, "CH4:1, O2:2, N2:7.52"
gas.equilibrate("HP")
print(f"Cantera {ct.__version__}: adiabatic flame T = {gas.T:.1f} K")

gas.TPX = 300, ct.one_atm, "CH4:1, O2:2, N2:7.52"
flame = ct.FreeFlame(gas, width=0.03)
flame.solve(loglevel=0, auto=True)
print(f"Laminar flame speed = {flame.velocity[0]:.3f} m/s")
