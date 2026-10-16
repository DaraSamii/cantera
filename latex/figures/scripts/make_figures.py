"""Generate every data figure used in the LaTeX tutorial.

Run from the repository root:
    .venv/bin/python latex/figures/scripts/make_figures.py [name ...]

With no arguments all figures are produced. Output: latex/figures/generated/*.pdf

The demonstrations deliberately use methane (GRI-Mech 3.0) or conditions that
differ from the H2/air project tasks, so that the project remains yours to do.
"""
import sys
from pathlib import Path

import cantera as ct
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "generated"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.figsize": (6.0, 3.8),
    "savefig.bbox": "tight",
    "lines.linewidth": 1.6,
})

AIR = "O2:1, N2:3.76"
P0 = ct.one_atm


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {name}.pdf")


# ---------------------------------------------------------------- thermo
def fig_nasa_cp():
    gas = ct.Solution("gri30.yaml")
    T = np.linspace(300, 3000, 400)
    names = ["H2", "O2", "N2", "H2O", "CO2", "OH"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for n in names:
        sp = gas.species(n)
        cp = np.array([sp.thermo.cp(t) for t in T]) / ct.gas_constant
        h = np.array([sp.thermo.h(t) - sp.thermo.h(298.15) for t in T]) / 1e6
        axes[0].plot(T, cp, label=n)
        axes[1].plot(T, h, label=n)
    for ax in axes:
        ax.axvline(1000, color="k", ls=":", lw=1)
        ax.set_xlabel("Temperature $T$ [K]")
    axes[0].set_ylabel(r"$c_{p,k}^\circ/R$ [-]")
    axes[0].text(1020, 2.6, "NASA range\nswitch, 1000 K", fontsize=8)
    axes[1].set_ylabel(r"$h_k^\circ(T)-h_k^\circ(298)$ [MJ/kmol]")
    axes[0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    save(fig, "nasa_cp")


def fig_gibbs_min():
    """G(xi) for 1 mol H2O <-> H2 + 1/2 O2 at fixed T, P (only these species)."""
    gas = ct.Solution("h2o2.yaml")
    iH2O, iH2, iO2 = (gas.species_index(s) for s in ("H2O", "H2", "O2"))
    xi = np.linspace(1e-6, 0.999, 2000)
    fig, ax = plt.subplots()
    for T in (2500, 3000, 3500):
        gas.TP = T, P0
        g = gas.standard_gibbs_RT  # g_k^o / RT
        n = np.vstack([1 - xi, xi, xi / 2])
        ntot = n.sum(axis=0)
        X = n / ntot
        G_RT = (n[0] * (g[iH2O] + np.log(X[0])) + n[1] * (g[iH2] + np.log(X[1]))
                + n[2] * (g[iO2] + np.log(X[2])))
        G_RT -= G_RT[0]
        line, = ax.plot(xi, G_RT, label=f"T = {T} K")
        k = np.argmin(G_RT)
        ax.plot(xi[k], G_RT[k], "o", color=line.get_color())
        # cross-check with Cantera restricted to the three species
        sub = ct.Solution(thermo="ideal-gas", species=[gas.species(s) for s in ("H2O", "H2", "O2")])
        sub.TPX = T, P0, "H2O:1"
        sub.equilibrate("TP")
        xH2 = sub.X[sub.species_index("H2")]
        xi_eq = 2 * xH2 / (2 - xH2) * 1  # from X_H2 = xi / (1 + xi/2)
        ax.axvline(xi_eq, color=line.get_color(), ls=":", lw=1)
    ax.set_xlabel(r"Extent of dissociation $\xi$ (mol H$_2$O dissociated)")
    ax.set_ylabel(r"$[G(\xi)-G(0)]/RT$ [-]")
    ax.set_title(r"H$_2$O $\rightleftharpoons$ H$_2$ + $\frac{1}{2}$O$_2$ at 1 atm: equilibrium = minimum of $G$")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(-0.7, 0.4)
    ax.legend()
    save(fig, "gibbs_min")


COMPLETE = ["CH4", "O2", "N2", "CO2", "H2O"]           # nothing can dissociate
MAJOR = ["CH4", "O2", "N2", "CO2", "H2O", "CO", "H2"]  # molecular dissociation, no radicals


def _tad_major_only(phi, T_u=300.0, names=COMPLETE):
    full = ct.Solution("gri30.yaml")
    sub = ct.Solution(thermo="ideal-gas", species=[full.species(s) for s in names])
    sub.set_equivalence_ratio(phi, "CH4", AIR)
    sub.TP = T_u, P0
    sub.equilibrate("HP")
    return sub.T


def fig_tad_phi_ch4():
    gas = ct.Solution("gri30.yaml")
    phi = np.linspace(0.4, 2.5, 85)
    T_full = []
    X = {s: [] for s in ("CO2", "H2O", "CO", "H2", "O2", "OH", "H", "O", "NO")}
    for p in phi:
        gas.set_equivalence_ratio(p, "CH4", AIR)
        gas.TP = 300, P0
        gas.equilibrate("HP")
        T_full.append(gas.T)
        for s in X:
            X[s].append(gas.X[gas.species_index(s)])
    T_complete = [_tad_major_only(p) for p in phi]
    T_major = [_tad_major_only(p, names=MAJOR) for p in phi]
    T_full = np.array(T_full)
    fig, ax = plt.subplots()
    ax.plot(phi, T_complete, ":", label=r"complete combustion (CO$_2$, H$_2$O only)")
    ax.plot(phi, T_major, "--", label="+ CO, H$_2$ (molecular dissociation)")
    ax.plot(phi, T_full, label="full equilibrium, 53 species (+ radicals)")
    k = np.argmax(T_full)
    ax.plot(phi[k], T_full[k], "ko")
    ax.annotate(fr"peak {T_full[k]:.0f} K at $\phi$ = {phi[k]:.2f}",
                (phi[k], T_full[k]), (1.4, 2300), arrowprops=dict(arrowstyle="->"))
    ax.axvline(1, color="k", ls=":", lw=1)
    ax.set_xlabel(r"Equivalence ratio $\phi$ [-]")
    ax.set_ylabel(r"$T_{ad}$ [K]")
    ax.set_title(r"CH$_4$/air, $T_u$ = 300 K, $p$ = 1 atm")
    ax.legend(fontsize=8, loc="lower center")
    save(fig, "tad_phi_ch4")

    fig, ax = plt.subplots()
    for s, v in X.items():
        ax.semilogy(phi, v, label=s)
    ax.axvline(1, color="k", ls=":", lw=1)
    ax.set_ylim(1e-6, 0.3)
    ax.set_xlabel(r"Equivalence ratio $\phi$ [-]")
    ax.set_ylabel("Equilibrium mole fraction [-]")
    ax.set_title(r"CH$_4$/air HP-equilibrium products ($T_u$ = 300 K, 1 atm)")
    ax.legend(ncol=3, fontsize=8)
    save(fig, "eq_comp_phi_ch4")


def fig_dissociation_TP():
    """Pure water vapour held at fixed T and P: dissociation vs T for 3 pressures."""
    gas = ct.Solution("h2o2.yaml")
    T = np.linspace(2000, 4000, 80)
    fig, ax = plt.subplots()
    for p_atm, ls in ((0.1, ":"), (1, "-"), (10, "--")):
        xh2o, xoh, xh = [], [], []
        for t in T:
            gas.TPX = t, p_atm * P0, "H2O:1"
            gas.equilibrate("TP")
            xh2o.append(gas.X[gas.species_index("H2O")])
            xoh.append(gas.X[gas.species_index("OH")])
            xh.append(gas.X[gas.species_index("H")])
        ax.plot(T, xh2o, "C0" + ls, label=f"H$_2$O, {p_atm} atm")
        ax.plot(T, xoh, "C1" + ls, label=f"OH, {p_atm} atm")
        ax.plot(T, xh, "C2" + ls, label=f"H, {p_atm} atm")
    ax.set_xlabel("Temperature [K] (held fixed)")
    ax.set_ylabel("Mole fraction [-]")
    ax.set_title("Dissociation of pure water vapour: high T and low p favour fragments")
    ax.legend(ncol=3, fontsize=7)
    save(fig, "dissociation_TP")


# ---------------------------------------------------------------- kinetics
def fig_arrhenius_crossover():
    gas = ct.Solution("h2o2.yaml")
    r_branch = gas.reaction(10)  # H + O2 <=> O + OH
    r_term = gas.reaction(8)     # H + O2 + N2 <=> HO2 + N2
    T = np.linspace(600, 2000, 300)
    k_b = np.array([r_branch.rate(t) for t in T])  # m3/kmol/s
    fig, ax = plt.subplots()
    ax.semilogy(1000 / T, 2 * k_b, "k", label=r"$2k_{\mathrm{br}}$: H + O$_2$ $\to$ O + OH")
    for p_atm, ls in ((1, "-"), (10, "--"), (50, ":")):
        M = p_atm * P0 / (ct.gas_constant * T)  # kmol/m3
        k_t = np.array([r_term.rate(t) for t in T]) * M
        ax.semilogy(1000 / T, k_t, "C3" + ls, label=fr"$k_{{\mathrm{{t}}}}[M]$: H + O$_2$ + M $\to$ HO$_2$ + M, {p_atm} atm")
        i = np.argmin(abs(np.log(2 * k_b) - np.log(k_t)))
        ax.plot(1000 / T[i], k_t[i], "o", color="C3")
        ax.annotate(f"{T[i]:.0f} K", (1000 / T[i], k_t[i]), (5, -14), textcoords="offset points", fontsize=8)
    ax.set_xlabel(r"$1000/T$ [1/K]")
    ax.set_ylabel(r"effective rate coefficient [m$^3$/(kmol s)]")
    ax.set_title("Chain branching vs. termination competing for H atoms")
    ax.legend(fontsize=7, loc="lower left")
    save(fig, "arrhenius_crossover")


def _ignite(T0, p, phi=1.0, mech="h2o2.yaml", fuel="H2", t_end=0.05):
    gas = ct.Solution(mech)
    gas.set_equivalence_ratio(phi, fuel, AIR)
    gas.TP = T0, p
    r = ct.IdealGasConstPressureReactor(gas, clone=True)
    net = ct.ReactorNet([r])
    states = ct.SolutionArray(gas, extra=["t"])
    t = 0.0
    while t < t_end:
        t = net.step()
        states.append(r.phase.state, t=t)
    dTdt = np.gradient(states.T, states.t)
    return states, states.t[np.argmax(dTdt)]


def fig_ignition_0D():
    states, tau = _ignite(1000.0, P0, t_end=2e-3)
    fig, axes = plt.subplots(2, 1, figsize=(6, 5.2), sharex=True)
    axes[0].plot(states.t * 1e3, states.T, "k")
    axes[0].axvline(tau * 1e3, color="r", ls="--", lw=1)
    axes[0].text(tau * 1e3 * 1.03, 1300, fr"$\tau_{{ign}}$ = {tau*1e3:.3f} ms", color="r")
    axes[0].set_ylabel("T [K]")
    axes[0].set_title(r"Constant-pressure ignition, H$_2$/air $\phi$=1, $T_0$=1000 K, 1 atm")
    for s in ("H2", "O2", "H2O", "OH", "H", "O", "HO2", "H2O2"):
        axes[1].semilogy(states.t * 1e3, states.X[:, states.species_index(s)], label=s)
    axes[1].set_ylim(1e-8, 1)
    axes[1].axvline(tau * 1e3, color="r", ls="--", lw=1)
    axes[1].set_xlabel("time [ms]")
    axes[1].set_ylabel("mole fraction")
    axes[1].legend(ncol=4, fontsize=7)
    axes[1].set_xlim(0, min(2.0, 3 * tau * 1e3))
    fig.tight_layout()
    save(fig, "ignition_0D")


def fig_ignition_delay():
    T0s = np.linspace(850, 1500, 18)
    fig, ax = plt.subplots()
    for p_atm, mk in ((1, "o-"), (10, "s--")):
        taus = []
        for T0 in T0s:
            _, tau = _ignite(T0, p_atm * P0, t_end=0.5 if T0 < 950 else 0.05)
            taus.append(tau)
        ax.semilogy(1000 / T0s, np.array(taus) * 1e3, mk, ms=3, label=f"{p_atm} atm")
    ax.set_xlabel(r"$1000/T_0$ [1/K]")
    ax.set_ylabel(r"$\tau_{ign}$ [ms]")
    ax.set_title(r"H$_2$/air ($\phi$=1) ignition delay: non-Arrhenius kink near crossover")
    ax.legend()
    save(fig, "ignition_delay")


# ---------------------------------------------------------------- transport
def fig_transport_T():
    gas = ct.Solution("gri30.yaml")
    T = np.linspace(300, 2500, 100)
    mu, lam, D, alpha = [], [], [], []
    for t in T:
        gas.TPX = t, P0, AIR
        mu.append(gas.viscosity)
        lam.append(gas.thermal_conductivity)
        alpha.append(gas.thermal_conductivity / (gas.density * gas.cp_mass))
        D.append(gas.mix_diff_coeffs_mass[gas.species_index("O2")])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    axes[0].loglog(T, np.array(mu) * 1e5, label=r"$\mu$ [10$^{-5}$ Pa s]")
    axes[0].loglog(T, np.array(lam) * 1e2, label=r"$\lambda$ [10$^{-2}$ W/m/K]")
    axes[0].loglog(T, 1.85 * (T / 300) ** 0.7, "k:", lw=1, label=r"$\propto T^{0.7}$")
    axes[1].loglog(T, np.array(alpha) * 1e4, label=r"$\alpha=\lambda/\rho c_p$")
    axes[1].loglog(T, np.array(D) * 1e4, label=r"$D_{\mathrm{O_2,mix}}$")
    axes[1].loglog(T, alpha[0] * 1e4 * (T / 300) ** 1.7, "k:", lw=1, label=r"$\propto T^{1.7}$")
    axes[1].set_ylabel(r"diffusivity [cm$^2$/s]")
    for ax in axes:
        ax.set_xlabel("T [K]")
        ax.legend(fontsize=8)
    axes[0].set_title("Air at 1 atm")
    fig.tight_layout()
    save(fig, "transport_T")


def fig_lewis():
    fig, ax = plt.subplots(figsize=(7, 3.3))
    for i, (mech, fuel) in enumerate((("gri30.yaml", "CH4"), ("h2o2.yaml", "H2"))):
        gas = ct.Solution(mech)
        gas.set_equivalence_ratio(1.0, fuel, AIR)
        gas.TP = 300, P0
        sp = ["H2", "H", "O2", "OH", "H2O", "N2"] + (["CH4", "CO", "CO2"] if fuel == "CH4" else [])
        a = gas.thermal_conductivity / (gas.density * gas.cp_mass)
        D = gas.mix_diff_coeffs_mass
        Le = [a / D[gas.species_index(s)] for s in sp]
        x = np.arange(len(sp)) + 0.4 * i
        ax.bar(x, Le, 0.4, label=f"in stoichiometric {fuel}/air, 300 K")
        if i == 0:
            ax.set_xticks(np.arange(len(sp)) + 0.2, sp)
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set_ylabel(r"$Le_k=\alpha/D_{k,m}$")
    ax.set_title("Species Lewis numbers in fresh gas: light species diffuse faster than heat")
    ax.legend(fontsize=8)
    save(fig, "lewis")


# ---------------------------------------------------------------- premixed
def _free_flame(phi, mech="gri30.yaml", fuel="CH4", T_u=300.0, p=P0, width=0.03,
                ratio=3, slope=0.06, curve=0.12, transport="mixture-averaged"):
    gas = ct.Solution(mech)
    gas.set_equivalence_ratio(phi, fuel, AIR)
    gas.TP = T_u, p
    f = ct.FreeFlame(gas, width=width)
    f.transport_model = transport
    f.set_refine_criteria(ratio=ratio, slope=slope, curve=curve)
    f.set_max_grid_points(1, 20000)
    f.solve(loglevel=0, auto=True)
    return f


def fig_freeflame_ch4():
    f = _free_flame(1.0, slope=0.03, curve=0.06)
    x = (f.grid - f.grid[np.argmax(f.heat_release_rate)]) * 1e3
    dTdx = np.gradient(f.T, f.grid)
    dL = (f.T[-1] - f.T[0]) / dTdx.max()
    fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    axes[0].plot(x, f.T, "k")
    axes[0].set_ylabel("T [K]")
    axes[0].set_title(fr"Freely propagating CH$_4$/air flame, $\phi$=1: $S_L$ = {f.velocity[0]*100:.1f} cm/s, $\delta_L$ = {dL*1e3:.2f} mm")
    ax2 = axes[0].twinx()
    ax2.plot(x, f.velocity, "C0--")
    ax2.set_ylabel("u [m/s]", color="C0")
    ax2.grid(False)
    for s in ("CH4", "O2", "H2O", "CO2", "CO"):
        axes[1].plot(x, f.Y[f.gas.species_index(s)], label=s)
    axes[1].set_ylabel("mass fraction")
    axes[1].legend(ncol=5, fontsize=8)
    for s in ("OH", "H", "O", "CH2O", "HO2"):
        axes[2].semilogy(x, f.X[f.gas.species_index(s)], label=s)
    axes[2].set_ylim(1e-6, 2e-2)
    ax3 = axes[2].twinx()
    ax3.fill_between(x, f.heat_release_rate / 1e6, color="C3", alpha=0.25)
    ax3.set_ylabel("HRR [MW/m$^3$]", color="C3")
    ax3.grid(False)
    axes[2].set_ylabel("mole fraction")
    axes[2].legend(ncol=5, fontsize=8, loc="upper left")
    axes[2].set_xlabel("x relative to heat-release peak [mm]")
    axes[2].set_xlim(-2, 4)
    for ax in axes:
        ax.axvspan(-2, -dL * 1e3 / 2, color="C0", alpha=0.06)
    axes[0].text(-1.9, 2000, "preheat zone", fontsize=8)
    axes[0].text(1.5, 1200, "post-flame / oxidation zone", fontsize=8)
    fig.tight_layout()
    save(fig, "freeflame_ch4")


def fig_SL_phi_ch4():
    phis = np.linspace(0.6, 1.5, 13)
    SL, dL, Tad = [], [], []
    f = None
    for phi in phis:
        f = _free_flame(phi)
        SL.append(f.velocity[0])
        dL.append((f.T[-1] - f.T[0]) / np.gradient(f.T, f.grid).max())
        g = ct.Solution("gri30.yaml")
        g.set_equivalence_ratio(phi, "CH4", AIR)
        g.TP = 300, P0
        g.equilibrate("HP")
        Tad.append(g.T)
        print(f"    phi={phi:.2f} SL={SL[-1]*100:.1f} cm/s")
    SL, dL = np.array(SL), np.array(dL)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    axes[0].plot(phis, SL * 100, "o-", ms=3)
    axes[0].set_ylabel(r"$S_L$ [cm/s]")
    ax2 = axes[0].twinx()
    ax2.plot(phis, Tad, "C3--")
    ax2.set_ylabel(r"$T_{ad}$ [K]", color="C3")
    ax2.grid(False)
    axes[1].plot(phis, dL * 1e3, "o-", ms=3, label=r"$\delta_L$ [mm]")
    gas = ct.Solution("gri30.yaml")
    gas.set_equivalence_ratio(1.0, "CH4", AIR)
    gas.TP = 300, P0
    a_u = gas.thermal_conductivity / (gas.density * gas.cp_mass)
    axes[1].plot(phis, a_u / SL * 1e3, "s--", ms=3, label=r"$\alpha_u/S_L$ [mm]")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel(r"$\phi$ [-]")
    axes[0].set_title(r"CH$_4$/air, 300 K, 1 atm")
    fig.tight_layout()
    save(fig, "SL_phi_ch4")


def fig_grid_convergence():
    """Lean CH4 (phi=0.6) flame solved with progressively tighter refinement."""
    crit = [(10, 0.9, 0.9), (5, 0.5, 0.5), (3, 0.3, 0.3), (3, 0.1, 0.2), (2, 0.05, 0.1), (2, 0.02, 0.04), (2, 0.01, 0.02)]
    g = ct.Solution("gri30.yaml")
    g.set_equivalence_ratio(0.6, "CH4", AIR)
    g.TP = 300, P0
    g.equilibrate("HP")
    T_eq = g.T
    N, SL, err = [], [], []
    for ratio, slope, curve in crit:
        f = _free_flame(0.6, ratio=ratio, slope=slope, curve=curve, width=0.05)
        N.append(len(f.grid))
        SL.append(f.velocity[0] * 100)
        err.append(abs(f.T[-1] - T_eq) / T_eq * 100)
        print(f"    N={N[-1]} SL={SL[-1]:.2f} cm/s Terr={err[-1]:.3f}%")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    axes[0].semilogx(N, SL, "o-")
    axes[0].set_ylabel(r"$S_L$ [cm/s]")
    axes[1].loglog(N, np.maximum(err, 1e-4), "o-")
    axes[1].axhline(1, color="r", ls="--", lw=1)
    axes[1].text(N[0], 1.15, "1 % warning level", color="r", fontsize=8)
    axes[1].set_ylabel(r"$|T_{end}-T_{eq}|/T_{eq}$ [%]")
    for ax in axes:
        ax.set_xlabel("number of grid points")
    axes[0].set_title(r"Grid study, CH$_4$/air $\phi$ = 0.6")
    fig.tight_layout()
    save(fig, "grid_convergence")


# ---------------------------------------------------------------- non-premixed
def _counterflow(mdot, width=0.02):
    gas = ct.Solution("gri30.yaml")
    gas.TP = 300, P0
    f = ct.CounterflowDiffusionFlame(gas, width=width)
    f.fuel_inlet.mdot = mdot
    f.fuel_inlet.X = "CH4:1"
    f.fuel_inlet.T = 300
    f.oxidizer_inlet.mdot = mdot * 3.0
    f.oxidizer_inlet.X = AIR
    f.oxidizer_inlet.T = 300
    f.set_refine_criteria(ratio=3, slope=0.1, curve=0.2, prune=0.03)
    f.solve(loglevel=0, auto=True)
    return f


def _Z_chi(f):
    Z = f.mixture_fraction("Bilger") if callable(f.mixture_fraction) else f.mixture_fraction
    Z = np.asarray(Z)
    alpha = f.thermal_conductivity / (f.density * f.cp)
    chi = 2 * alpha * np.gradient(Z, f.grid) ** 2
    return Z, chi


def _eq_T_of_Z(Z):
    """Equilibrium (HP) temperature of the adiabatic mixture of the two streams."""
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 300, P0, "CH4:1"
    hF, YF = gas.enthalpy_mass, gas.Y.copy()
    gas.TPX = 300, P0, AIR
    hO, YO = gas.enthalpy_mass, gas.Y.copy()
    T = []
    for z in Z:
        gas.HPY = z * hF + (1 - z) * hO, P0, z * YF + (1 - z) * YO
        gas.equilibrate("HP")
        T.append(gas.T)
    return np.array(T)


def _Zst():
    gas = ct.Solution("gri30.yaml")
    gas.set_equivalence_ratio(1.0, "CH4", AIR)
    return gas.mixture_fraction("CH4:1", AIR)


def fig_counterflow_ch4():
    Zst = _Zst()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    for mdot in (0.05, 0.2, 0.45):
        f = _counterflow(mdot)
        Z, chi = _Z_chi(f)
        ist = np.argmin(abs(Z - Zst))
        a = f.strain_rate("max")
        axes[0].plot(f.grid * 1e3, f.T, label=fr"$a_{{max}}$ = {a:.0f} 1/s")
        axes[1].plot(Z, f.T, label=fr"$\chi_{{st}}$ = {chi[ist]:.2f} 1/s")
    Zg = np.concatenate([np.linspace(0, 0.2, 60), np.linspace(0.2, 1, 25)])
    axes[1].plot(Zg, _eq_T_of_Z(Zg), "k--", lw=1.2, label="equilibrium $T_{eq}(Z)$")
    # Burke-Schumann: complete combustion, T piecewise linear between (0,300),(Zst,Tst),(1,300)
    g = ct.Solution("gri30.yaml")
    g.set_equivalence_ratio(1.0, "CH4", AIR)
    g.TP = 300, P0
    Tst = _tad_major_only(1.0)
    axes[1].plot([0, Zst, 1], [300, Tst, 300], "k:", lw=1.2, label="Burke-Schumann (no dissoc.)")
    axes[1].axvline(Zst, color="gray", lw=0.8)
    axes[1].set_xlim(0, 0.3)
    axes[0].set_xlabel("x [mm] (fuel inlet at 0)")
    axes[0].set_ylabel("T [K]")
    axes[1].set_xlabel("mixture fraction Z [-]")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    axes[0].set_title(r"CH$_4$ vs air counterflow")
    axes[1].set_title(r"Same flames in Z-space ($Z_{st}$ marked)")
    fig.tight_layout()
    save(fig, "counterflow_ch4")


def fig_extinction_ch4():
    """Upper branch of the S-curve: T_max vs chi_st, increasing strain until extinction."""
    Zst = _Zst()
    gas = ct.Solution("gri30.yaml")
    gas.TP = 300, P0
    f = ct.CounterflowDiffusionFlame(gas, width=0.02)
    mdot = 0.2
    f.fuel_inlet.mdot, f.fuel_inlet.X, f.fuel_inlet.T = mdot, "CH4:1", 300
    f.oxidizer_inlet.mdot, f.oxidizer_inlet.X, f.oxidizer_inlet.T = 3 * mdot, AIR, 300
    f.set_refine_criteria(ratio=3, slope=0.1, curve=0.2, prune=0.03)
    f.solve(loglevel=0, auto=True)
    chis, Tmax = [], []
    alpha = 1.25
    backup = OUT / "_extinction_backup.yaml"

    def record():
        Z, chi = _Z_chi(f)
        chis.append(chi[np.argmin(abs(Z - Zst))])
        Tmax.append(f.T.max())
        f.save(backup, name="last", overwrite=True)
        print(f"    mdot={mdot:.3f} chi_st={chis[-1]:.2f} Tmax={Tmax[-1]:.0f}")

    record()
    while alpha - 1 > 2e-3 and len(chis) < 120:
        # similarity scaling of the previous solution (Fiala & Sattelmayer):
        # strain ~ alpha, grid ~ alpha^-1/2, u ~ alpha^1/2, V ~ alpha, Lambda ~ alpha^2
        x = f.grid
        xn = (x - x[0]) / (x[-1] - x[0])
        u, V, L = f.velocity * alpha ** 0.5, f.spread_rate * alpha, f.L * alpha ** 2
        f.flame.grid = x * alpha ** -0.5
        f.fuel_inlet.mdot = mdot * alpha ** 0.5
        f.oxidizer_inlet.mdot = 3 * mdot * alpha ** 0.5
        f.set_profile("velocity", xn, u)
        f.set_profile("spreadRate", xn, V)
        f.set_profile("Lambda", xn, L)
        try:
            f.solve(loglevel=0, auto=False)
            burning = f.T.max() > 900
        except ct.CanteraError:
            burning = False
        if burning:
            mdot *= alpha ** 0.5
            record()
        else:  # went past extinction: step back and take a smaller step
            f.restore(backup, name="last")
            alpha = 1 + (alpha - 1) / 2
    backup.unlink(missing_ok=True)
    fig, ax = plt.subplots()
    ax.semilogx(chis, Tmax, "o-", ms=3, label="computed (upper, burning branch)")
    ax.axhline(300, color="k", lw=1, label="lower, frozen branch")
    ax.axvline(chis[-1], color="r", ls="--", lw=1)
    ax.text(chis[-1] * 0.9, 1000, fr"extinction $\chi_q\approx$ {chis[-1]:.0f} 1/s", color="r", ha="right")
    ax.set_xlabel(r"$\chi_{st}$ [1/s]")
    ax.set_ylabel(r"$T_{max}$ [K]")
    ax.set_title(r"CH$_4$/air counterflow: approach to extinction")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "extinction_ch4")


FIGS = {name[4:]: fn for name, fn in dict(globals()).items() if name.startswith("fig_")}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(FIGS)
    for name in todo:
        print(f"[{name}]")
        FIGS[name]()
