"""Utility functions for working with materials."""

from __future__ import annotations

import csv
import warnings
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import yaml
from matplotlib.axes import Axes
from scipy.cluster.vq import kmeans2

import optiland.backend as be
from optiland.materials.material import Material

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def glasses_selection(
    lambda_min: float,
    lambda_max: float,
    catalogs: list[str] | None = None,
) -> list[str]:
    """
    Retrieves a list of glass names whose tabulated transmission window fully covers
    the specified wavelength interval [lambda1, lambda2].
    Optionally filters by catalog names.

    Args:
        lambda_min (float): The lower wavelength bound in microns.
        lambda_max (float): The upper wavelength bound in microns.
        catalogs (list[str], optional): List of catalog names.
            Catalog names are case-insensitive.
            Defaults to None.

    Returns:
        list[str]: List of unique glass names transmissive within
                   the wavelength interval and matching catalogs $
                   if specified.
    """

    csv_path = resources.files("optiland.database").joinpath("catalog_nk.csv")
    if catalogs is not None:
        catalogs = [c.lower() for c in catalogs]
    glasses = set()

    with open(csv_path, encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=",")
        for row in reader:
            group = row["group"].lower()
            catalog = row["filename"].split("/")[1]
            try:
                min_wavelength = float(row["min_wavelength"])
                max_wavelength = float(row["max_wavelength"])
            except ValueError:
                continue  # Skip rows with invalid wavelength values

            # Check if it's a glass, within wavelength range,
            # and optionally matches catalog
            if (
                "glass" in group
                and min_wavelength <= lambda_min
                and max_wavelength >= lambda_max
                and (catalogs is None or catalog in catalogs)
            ):
                glasses.add(row["filename_no_ext"])

    return sorted(glasses)


def get_nd_vd(glass: str) -> tuple[float, float]:
    """
    Retrieve the refractive index (n_d) and Abbe number (V_d) for a given glass.

    This function loads the material data associated with the specified glass name,
    reads its YAML specification file, and extracts the n_d and V_d values from the
    "SPECS" section. If either value is missing, defaults to (0, 0).

    Args:
        glass (str): Name of the glass material.

    Returns:
        tuple[float, float]: A tuple (n_d, V_d) representing
        the refractive index and Abbe number, in the form:
        glass_dict = {'S-BSM22': (1.62, 53.16), 'LF5G19': (1.59, 39.89), ...}
    """
    material = Material(glass)
    yml_path, _ = material._retrieve_file()
    with Path(yml_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    try:
        nd = data["SPECS"]["nd"]
        vd = data["SPECS"]["Vd"]
    except KeyError:
        nd, vd = 0, 0
    return nd, vd


def downsample_glass_map(glass_dict: dict, num_glasses_to_keep: int) -> dict:
    """
    Downsample a glass dictionary using K-Means clustering in the (n_d, V_d) space.

    The dictionary must be in the form:
        glass_dict = {'S-BSM22': (1.62, 53.16), 'LF5G19': (1.59, 39.89), ...}

    Args:
        glass_dict (dict): Dictionary with glass names as keys
                           and (n_d, V_d) tuples as values.
        num_glasses_to_keep (int): Number of representative glasses
                                   to retain after clustering.

    Returns:
        dict: Downsampled dictionary with selected representative glasses.

    Raises:
        ValueError: If num_glasses_to_keep is greater
                    than the number of available glasses.
    """
    # Validate input
    assert num_glasses_to_keep <= len(glass_dict), (
        "Cannot keep more glasses than available in the input dictionary."
    )
    assert num_glasses_to_keep > 1, "Must retain at least 2 glasses."

    # Extract glass names and (n_d, V_d) data
    glass_names = list(glass_dict.keys())
    glass_data = be.array([glass_dict[name] for name in glass_names])

    # Detach if PyTorch and convert to NumPy for SciPy clustering
    if be.get_backend() == "torch":
        glass_data = glass_data.detach().cpu().numpy()

    # Perform K-Means clustering
    centroids, labels = kmeans2(
        glass_data, num_glasses_to_keep, minit="points", seed=1234
    )

    # Warn if kmeans2 produced fewer clusters than requested
    num_unique_clusters = len(set(labels))
    if num_unique_clusters < num_glasses_to_keep:
        warnings.warn(
            f"In downsample_glass_map(): K-Means produced only "
            f"{num_unique_clusters} clusters out of "
            f"{num_glasses_to_keep} requested. "
            f"Some clusters may be empty and "
            f"fewer glasses will be selected.",
            category=UserWarning,
            stacklevel=2,
        )

    # Select the closest glass to each centroid
    selected_glasses = {}
    labels = be.array(labels)
    for cluster_index in range(num_glasses_to_keep):
        # Get indices of glasses in this cluster
        mask = labels == cluster_index
        indices = be.arange_indices(be.size(labels))
        cluster_indices = indices[mask]

        # Extract cluster points
        cluster_points = be.array(glass_data[cluster_indices])
        centroid = be.array(centroids[cluster_index])

        # Ensure cluster_points and centroid are 2D in torch
        if be.get_backend() == "torch":
            if cluster_points.ndim == 1:
                cluster_points = cluster_points.unsqueeze(0)
            if centroid.ndim == 1:
                centroid = centroid.unsqueeze(0)
            # Compute distances
            delta = cluster_points - centroid
            distances = be.linalg.norm(delta, dim=1 if delta.ndim > 1 else 0)
            closest_local_idx = be.argmin(distances).item()
        else:
            # NumPy distances
            distances = be.linalg.norm(cluster_points - centroid, axis=1)
            closest_local_idx = be.argmin(distances)

        # Get global index of closest point
        closest_index = cluster_indices[closest_local_idx]
        glass_name = glass_names[closest_index]
        selected_glasses[glass_name] = glass_dict[glass_name]

    return selected_glasses


def get_neighbour_glasses(
    glass: str,
    glass_selection: list[str] = None,
    glass_dict: dict[str, tuple[float, float]] = None,
    num_neighbours: int = 3,
    plot: bool = False,
) -> list[str]:
    """
    Return the `num_neighbours` closest glasses
    to the given `glass` in (n_d, V_d) space.

    Args:
        glass (str): Name of the reference glass.
        glass_selection (list[str]): List of glass names to search among.
        glass_dict (dict[str, tuple[float, float]]): dict of glass names associated
            with their (nv,vd) pairs.
        num_neighbours (int): Number of closest glasses to return.
        plot (bool): If True, plot the selected glass map
                     and highlight neighbors.

    Returns:
        list[str]: List of `num_neighbours` closest glasses
                   to `glass` in (n_d, V_d) space.
    """

    # If not provided, get index and dispersion (n_d, V_d) for all glasses
    if not glass_selection and not glass_dict:
        if glass not in glass_selection:
            raise ValueError(
                f"In get_neighbor_glasses(): glass {glass} "
                f"not found in selected glasses"
            )
        glass_dict = {g: get_nd_vd(g) for g in glass_selection}

    nd0, vd0 = glass_dict[glass]

    # Compute Euclidean distance in (nd, vd) space. Weighting can be added.
    # For each glass in the catalogue:
    # - Skip the current glass (it should not be a neighbor of itself)
    # - Calculate the Euclidean distance in (nd, vd) space
    #   between the candidate and the current glass
    # - Store the distance and the corresponding glass name
    #   in a list for sorting
    distances: list[tuple[float, str]] = []
    for g, (nd, vd) in glass_dict.items():
        if g == glass:
            continue
        w_nd, w_vd = 1.0, 1.0  # For future weighting
        distance = be.hypot(w_nd * (nd - nd0), w_vd * (vd - vd0))
        distances.append((distance, g))

    distances.sort(key=lambda t: t[0])
    neighbours: list[str] = [g for _, g in distances[:num_neighbours]]

    # Optional plotting
    if plot:
        plot_glass_map(
            glass_selection=list(glass_dict.keys()),
            highlights=neighbours,
            title="Glass map",
        )

    return neighbours


def plot_glass_map(
    glass_selection: list[str],
    highlights: list[str],
    title: str = None,
) -> None:
    """
    Plot a (n_d, V_d) scatter map of selected optical glasses,
    highlighting specific ones.

    This visualization is useful to understand the distribution
    of glasses in the (index, Abbe number) space and to inspect
    how selected or substituted glasses relate  to the rest of
    the catalogue.

    Args:
        glass_selection (list[str]): List of glass names to plot.
        highlights (list[str]): List of glass names to highlight
        in red (e.g. current neighbors).
        title (str, optional): Custom title for the plot.
        If not provided, a default title is used.

    Notes:
        - Glasses with invalid (0, 0) values for (n_d, V_d)
          are skipped and reported.
        - The V_d axis is reversed to match
          optical engineering conventions.
    """
    # Buffers for standard and highlighted glasses
    x_vd, y_nd, labels = [], [], []
    x_vd_hl, y_nd_hl, labels_hl = [], [], []

    # Retrieve (n_d, V_d) data for all selected glasses
    glass_dict = {g: get_nd_vd(g) for g in glass_selection}

    for glass_name in glass_selection:
        nd, vd = glass_dict.get(glass_name, (0, 0))

        # Skip glasses with missing or invalid data
        if (nd, vd) == (0, 0):
            # print(f"In plot_glass_map(): couldn't display "
            #       f"{glass_name}, invalid (n_d, V_d) data.")
            continue

        # Route to highlight or standard category
        if glass_name in highlights:
            x_vd_hl.append(vd)
            y_nd_hl.append(nd)
            labels_hl.append(glass_name)
        else:
            x_vd.append(vd)
            y_nd.append(nd)
            labels.append(glass_name)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all standard glasses
    ax.scatter(x_vd, y_nd, c="gray", marker="+", s=20, label="Glasses")
    for i, glass_name in enumerate(labels):
        ax.text(x_vd[i], y_nd[i], glass_name, fontsize=8, ha="right", va="bottom")

    # Plot highlighted glasses in red
    if x_vd_hl:
        ax.scatter(x_vd_hl, y_nd_hl, c="red", label="Highlighted")
        for i, glass_name in enumerate(labels_hl):
            ax.text(
                x_vd_hl[i],
                y_nd_hl[i],
                glass_name,
                fontsize=9,
                ha="left",
                va="top",
                color="red",
                fontweight="bold",
            )

    # Set axis labels and title
    ax.set_xlabel("$V_d$")
    ax.set_ylabel("$n_d$")
    ax.set_title("Glass Map: $n_d$ vs. $V_d$" if title is None else title)

    # Reverse x-axis (V_d) to match optical convention
    ax.invert_xaxis()
    ax.grid(alpha=0.5)

    # Alternate horizontal bands for readability
    fig.canvas.draw()
    yticks = ax.get_yticks()
    for i in range(1, len(yticks), 2):
        try:
            ax.axhspan(yticks[i], yticks[i + 1], facecolor="lightblue", alpha=0.15)
        except IndexError:
            break  # Last pair might not exist

    fig.tight_layout()
    return fig, ax


def find_closest_glass(nd_vd: tuple, catalog: list[str], plot_map: bool = False) -> str:
    """
    Find the glass in `catalog` whose (nd, vd) is closest to `nd_vd`.

    Args:
        nd_vd (tuple): Target (nd, vd).
        catalog (list[str]): List of glass names.

    Returns:
        str: Name of the closest glass.
    """
    # Create a mapping of glass name to (nd, vd)
    glass_dict = {g: get_nd_vd(g) for g in catalog}

    # Find the glass with the minimum squared Euclidean distance
    closest_glass = min(
        glass_dict.items(),
        key=lambda item: (item[1][0] - nd_vd[0]) ** 2 + (item[1][1] - nd_vd[1]) ** 2,
    )[0]

    if plot_map:
        plot_glass_map(
            glass_selection=catalog,
            highlights=closest_glass,
            title="Map",
        )

    return closest_glass


def plot_nk(
    material: Material,
    wavelength_range: tuple[float, float] | None = None,
    ax: Axes | tuple[Axes, Axes] | None = None,
    n_sample: int = 800,
    share_yscale: bool = False,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """
    Plot the refractive index (n) and extinction coefficient (k) of a material
    versus wavelength.

    Args:
        material (Material): The material object containing optical data.
        wavelength_range (tuple): The range of wavelengths to plot (min_wl, max_wl)
        in micrometers. If None, the full range is used. If wavelength_range is not
        contained within the material's range, red shading is applied to indicate
        the out-of-bounds region.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new
        figure and axes are created.
        n_sample (int): The number of wavelength samples to compute.
        share_yscale (bool): Whether to share the y-axis scale between n and k plots.

    Returns:
        A tuple containing the figure and a list of two axes objects

    Example:
    >>> mat = Material("BK7", reference="SCHOTT")
    >>> plot_nk(mat, wavelength_range=(0.4, 0.7))

    """

    # gather wavelength range information
    min_wl = material.material_data.get("min_wavelength")
    max_wl = material.material_data.get("max_wavelength")

    if min_wl is None or max_wl is None:
        raise ValueError(
            "Failed to fetch minimum and maximum wavelength from material."
        )

    # Check if the specified wavelength_range is valid
    if wavelength_range is None:
        wavelength_range = (min_wl, max_wl)
    if len(wavelength_range) != 2:
        raise ValueError("wavelength_range must be a tuple of (min_wl, max_wl)")

    if ax is None:
        fig, ax_n = plt.subplots()
        ax_k = ax_n.twinx()
    elif (
        isinstance(ax, tuple) and len(ax) == 2 and all(isinstance(a, Axes) for a in ax)
    ):
        ax_n, ax_k = ax
    elif isinstance(ax, Axes):
        ax_n = ax
        ax_k = ax.twinx()
    else:
        raise ValueError(
            f"Invalid ax argument should be None, a tuple of (ax_n, ax_k), or "
            f"a single Axes instance. Here : {type(ax)}"
        )

    # Check if the specified wavelength_range is valid
    if wavelength_range is None:
        raise ValueError("Wavelength range not initialized")
    if min_wl > wavelength_range[0] or max_wl < wavelength_range[1]:
        warnings.warn(
            "Specified wavelength_range is outside the material's available range. "
            "Red shading will indicate out-of-bounds regions.",
            UserWarning,
            stacklevel=2,
        )
        # Shade the out-of-bounds region
        if min_wl > wavelength_range[0]:
            ax_n.axvspan(
                min_wl,
                wavelength_range[0],
                facecolor="red",
                alpha=0.15,
                label="Out of bounds",
            )
        if max_wl < wavelength_range[1]:
            ax_n.axvspan(wavelength_range[1], max_wl, facecolor="red", alpha=0.15)

    # Plot n and k
    wl = be.linspace(*wavelength_range, n_sample)
    n = material.n(wl)
    k = material.k(wl)

    ax_n.plot(wl, n, label="n", color="k")
    ax_k.plot(wl, k, label="k", color="k", linestyle=":")
    ax_n.set_xlabel(r"$\lambda$ (nm)")
    ax_n.set_ylabel("$n$", color="k")
    ax_n.tick_params(axis="y", labelcolor="k")
    ax_k.set_ylabel("$k$", color="k")
    ax_k.tick_params(axis="y", labelcolor="k")

    # title stuff
    full_name = material.material_data.get("category_name_full", "")
    ref = material.material_data.get("reference", "")
    full_name = full_name.replace("<sub>", "$_{")
    full_name = full_name.replace("</sub>", "}$")
    ax_n.set_title(f"{full_name} - {ref}")

    ax_n.set_xlim(wavelength_range)
    ax_k.set_xlim(wavelength_range)

    if share_yscale:
        ymin = min(ax_n.get_ylim()[0], ax_k.get_ylim()[0])
        ymax = max(ax_n.get_ylim()[1], ax_k.get_ylim()[1])
        ax_n.set_ylim(ymin, ymax)
        ax_k.set_ylim(ymin, ymax)

    # Combine legends from both axes
    lines, labels = ax_n.get_legend_handles_labels()
    lines2, labels2 = ax_k.get_legend_handles_labels()
    ax_n.legend(lines + lines2, labels + labels2)

    return fig, (ax_n, ax_k)
