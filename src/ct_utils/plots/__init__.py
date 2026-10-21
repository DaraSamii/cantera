"""Plot style, plotting helpers and figure saving.

Call :func:`set_style` once at the top of a notebook. It sets LaTeX fonts
(real LaTeX when it is installed, Computer Modern mathtext otherwise) and a
colourblind-safe colour cycle (Okabe-Ito) paired with distinct markers, so
curves stay distinguishable in greyscale and for colour-vision deficiencies.
Colour maps default to ``viridis``, which is perceptually uniform and
colourblind-safe.
"""

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from ct_utils import figures_dir

#: Okabe-Ito colourblind-safe palette, ordered for contrast on white.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00",
           "#CC79A7", "#56B4E9", "#000000", "#F0E442"]

#: Markers paired one-to-one with PALETTE.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

#: Default colour map for maps and surfaces.
CMAP = "viridis"


def set_style(usetex=None, font_size=11):
    """
    Apply a professional, colourblind-safe Matplotlib style.

    Parameters
    ----------
    usetex : bool or None, optional
        Render text with LaTeX. ``None`` (default) uses LaTeX only if the
        ``latex`` and ``dvipng`` executables are found.
    font_size : float, optional
        Base font size in points. Default 11.

    Returns
    -------
    bool
        Whether LaTeX rendering is active.
    """
    if usetex is None:
        usetex = all(shutil.which(exe) for exe in ("latex", "dvipng"))

    mpl.rcParams.update({
        # Fonts
        "text.usetex": usetex,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "legend.fontsize": font_size - 2,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        # Colours and lines
        "axes.prop_cycle": cycler(color=PALETTE) + cycler(marker=MARKERS),
        "image.cmap": CMAP,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        # Axes and ticks
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "legend.frameon": False,
        # Output
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })
    return usetex


def _as_series(x, y):
    """Normalise x and y into two equal-length lists of 1D arrays."""
    several_y = isinstance(y, (list, tuple)) and np.ndim(y[0]) == 1
    ys = [np.asarray(v, dtype=float) for v in y] if several_y or np.ndim(y) == 2 \
        else [np.asarray(y, dtype=float)]
    several_x = isinstance(x, (list, tuple)) and np.ndim(x[0]) == 1
    xs = [np.asarray(v, dtype=float) for v in x] if several_x \
        else [np.asarray(x, dtype=float)] * len(ys)
    return xs, ys


def _reference_lines(ax, values, orientation):
    """Draw labelled dashed reference lines (vertical or horizontal)."""
    if values is None:
        return
    if not isinstance(values, dict):
        values = {v: None for v in np.atleast_1d(values)}
    draw = ax.axvline if orientation == "v" else ax.axhline
    for value, label in values.items():
        draw(value, color="0.45", ls="--", lw=1.0, label=label, zorder=0)


def plot(x, y, xlabel, ylabel, figsize=(6.4, 4.2), legend=True, grid=True,
         labels=None, title=None, ax=None, xscale="linear", yscale="linear",
         markevery=None, colors=None, vlines=None, hlines=None, **kwargs):
    """
    Plot one or several curves with consistent styling.

    Parameters
    ----------
    x : array-like or list of array-like
        Shared x values, or one x array per curve.
    y : array-like, list of array-like, or 2D array
        One curve, or several (one per list item or array row).
    xlabel, ylabel : str
        Axis labels, with units (LaTeX math allowed).
    figsize : tuple of float, optional
        Figure size in inches. Ignored when ``ax`` is given.
    legend : bool, optional
        Show a legend when there are labels. Default True.
    grid : bool, optional
        Show a dotted grid. Default True.
    labels : list of str, optional
        One label per curve.
    title : str, optional
        Axes title.
    ax : matplotlib.axes.Axes, optional
        Draw into this axes instead of creating a figure.
    xscale, yscale : str, optional
        Axis scales, e.g. ``"log"``. Default linear.
    markevery : int, optional
        Draw a marker every n points (useful for dense curves).
    colors : list of str, optional
        One colour per curve, overriding the default cycle.
    vlines, hlines : float, list of float, or dict of float to str, optional
        Dashed reference lines; a dict maps each value to its legend label.
    **kwargs
        Passed to ``Axes.plot`` for every curve.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    xs, ys = _as_series(x, y)
    labels = labels if labels is not None else [None] * len(ys)
    for k, (xk, yk, lab) in enumerate(zip(xs, ys, labels)):
        style = dict(kwargs)
        if colors is not None:
            style.setdefault("color", colors[k])
            style.setdefault("marker", MARKERS[k % len(MARKERS)])
        ax.plot(xk, yk, label=lab, markevery=markevery, **style)

    _reference_lines(ax, vlines, "v")
    _reference_lines(ax, hlines, "h")

    ax.set(xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, which="both")
    if legend and ax.get_legend_handles_labels()[1]:
        ax.legend()
    return fig, ax


def contour_map(x, y, Z, xlabel, ylabel, zlabel, figsize=(6.4, 4.6),
                levels=20, isolines=None, mark_max=True, ax=None,
                xscale="linear", title=None, cmap=CMAP):
    """
    Filled contour map of Z(x, y) with a colour bar.

    Parameters
    ----------
    x : array-like, shape (nx,)
        Column coordinates.
    y : array-like, shape (ny,)
        Row coordinates.
    Z : array-like, shape (ny, nx)
        Values on the grid.
    xlabel, ylabel, zlabel : str
        Axis and colour-bar labels.
    figsize : tuple of float, optional
        Figure size in inches. Ignored when ``ax`` is given.
    levels : int or array-like, optional
        Filled contour levels. Default 20.
    isolines : list of float, optional
        Values to draw as labelled black isolines.
    mark_max : bool, optional
        Mark the maximum with a star and a legend entry. Default True.
    ax : matplotlib.axes.Axes, optional
        Draw into this axes instead of creating a figure.
    xscale : str, optional
        x-axis scale. Default linear.
    title : str, optional
        Axes title.
    cmap : str, optional
        Colour map. Default ``viridis``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    X, Y = np.meshgrid(x, y)
    Z = np.asarray(Z, dtype=float)

    cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    fig.colorbar(cs, ax=ax, label=zlabel)

    if isolines is not None:
        cl = ax.contour(X, Y, Z, levels=sorted(isolines), colors="k", linewidths=1.2)
        ax.clabel(cl, fmt=lambda v: f"{v:.0f}", fontsize=8)

    if mark_max:
        i, j = np.unravel_index(np.argmax(Z), Z.shape)
        ax.plot(X[i, j], Y[i, j], marker="*", color="white", mec="k", ms=14, ls="none",
                label=f"max {Z[i, j]:.0f} at ({X[i, j]:.3g}, {Y[i, j]:.3g})")
        ax.legend(loc="upper right", frameon=True)

    ax.set(xlabel=xlabel, ylabel=ylabel, xscale=xscale)
    if title:
        ax.set_title(title)
    return fig, ax


def surface_3d(x, y, Z, xlabel, ylabel, zlabel, figsize=(7.0, 5.2),
               lines=None, mark_max=True, ax=None, view=(25, -130),
               title=None, cmap=CMAP):
    """
    3D surface of Z(x, y), optionally with curves drawn on top.

    Parameters
    ----------
    x : array-like, shape (nx,)
        Column coordinates.
    y : array-like, shape (ny,)
        Row coordinates.
    Z : array-like, shape (ny, nx)
        Surface heights.
    xlabel, ylabel, zlabel : str
        Axis labels.
    figsize : tuple of float, optional
        Figure size in inches. Ignored when ``ax`` is given.
    lines : list of dict, optional
        Curves to draw on the surface. Each dict needs keys ``x``, ``y`` and
        ``z`` (arrays) and may hold any ``Axes.plot`` option (e.g. ``color``).
    mark_max : bool, optional
        Mark the maximum with a point. Default True.
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        Draw into this 3D axes instead of creating a figure.
    view : tuple of float, optional
        (elevation, azimuth) of the camera in degrees.
    title : str, optional
        Axes title.
    cmap : str, optional
        Colour map. Default ``viridis``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : mpl_toolkits.mplot3d.Axes3D
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.figure
    X, Y = np.meshgrid(x, y)
    Z = np.asarray(Z, dtype=float)

    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="none",
                           alpha=0.6 if lines else 0.95)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.12, label=zlabel)

    lift = 0.005 * np.ptp(Z)        # keeps curves visible above the surface
    for line in lines or []:
        opts = {k: v for k, v in line.items() if k not in ("x", "y", "z")}
        opts.setdefault("lw", 2)
        opts.setdefault("marker", "none")
        ax.plot(line["x"], line["y"], np.asarray(line["z"]) + lift, **opts)

    if mark_max:
        i, j = np.unravel_index(np.argmax(Z), Z.shape)
        ax.scatter(X[i, j], Y[i, j], Z[i, j], color="white", edgecolor="k", s=50,
                   depthshade=False)

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.zaxis.set_rotate_label(False)        # keep the z label upright
    ax.set_zlabel(zlabel, rotation=90)
    ax.view_init(*view)
    if title:
        ax.set_title(title)
    return fig, ax


def save_figure(fig, name, formats=("pdf", "jpeg"), folder=None):
    """
    Save a figure once per format, each in its own subfolder.

    For the defaults this writes ``results/<part>/figures/pdf/<name>.pdf``
    and ``results/<part>/figures/jpeg/<name>.jpeg``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        File name without extension.
    formats : tuple of str, optional
        File formats. Default ``("pdf", "jpeg")``.
    folder : path-like, optional
        Parent folder. Default ``results/<part>/figures`` (see
        ``ct_utils.set_part``).

    Returns
    -------
    list of pathlib.Path
        Paths of the written files.
    """
    folder = Path(folder) if folder is not None else figures_dir()
    paths = []
    for fmt in formats:
        sub = folder / fmt
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / f"{name}.{fmt}"
        fig.savefig(path, format=fmt)
        paths.append(path)
    return paths
