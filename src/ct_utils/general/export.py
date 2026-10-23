"""Saving results as CSV in the current part's ``results/<part>/raw`` folder."""

from pathlib import Path

from ct_utils import raw_dir


# %============================================================================
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


# %============================================================================
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
