"""Thin film optics stack class with inlined TMM.

This class encapsulates both the stack structure (incident/substrate, layers)
and the numerical Transfer Matrix Method (TMM) to compute complex amplitude
coefficients (r, t) and power coefficients (R, T, A) for s, p and unpolarized
cases.

Corentin Nannini, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import optiland.backend as be
from optiland.materials import IdealMaterial

from .core import _tmm_coh
from .layer import Layer

if TYPE_CHECKING:
    from optiland.materials import BaseMaterial
import re

import matplotlib.pyplot as plt

Pol = Literal["s", "p", "u"]
PlotType = Literal["R", "T", "A"]
Array: TypeAlias = Any  # be.ndarray


def _to_float(value: Any) -> float:
    """Convert a scalar (possibly a torch tensor) to a Python float."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _material_display_name(obj: Any) -> str:
    """Human-readable label for a material or layer used in stack plots.

    IdealMaterial may not have a name, so its refractive index is used
    for labeling instead.
    """
    name = getattr(obj, "name", "") or ""
    if isinstance(obj, IdealMaterial):
        name = f"$n$ = {_to_float(obj.index[0])}"
    return name


@dataclass
class _StackBand:
    """One rectangle band (substrate, layer, or incident medium) to render."""

    y: float
    height: float
    color: str
    label: str
    text: str | None = None


@dataclass
class ThinFilmStack:
    """Multilayer thin-film stack with inlined TMM calculations.

    This class encapsulates both the stack structure (incident/substrate, layers)
    and the numerical Transfer Matrix Method (TMM) to compute complex amplitude
    coefficients (r, t) and power coefficients (R, T, A) for s, p and unpolarized
    cases.

    Units and conventions:
    - Wavelength in microns (µm) internally; convenience helpers accept nm.
    - AOI in radians internally; convenience helpers accept degrees.
    - Layers are ordered from the incident side to the substrate side.

    Args:
        incident_material (BaseMaterial): Incident medium (e.g., air).
        substrate_material (BaseMaterial): Substrate medium (e.g., glass).
        layers (list[Layer], optional): Ordered layers between incident and
            substrate. Defaults to None.
        reference_wl_um (float | None, optional): Reference wavelength for
            thickness quarter-wave calculations. Defaults to None.
        reference_AOI_deg (float | None, optional): Reference angle of
            incidence in degrees for thickness quarter-wave calculations.
            Defaults to 0 (normal incidence).

    Examples:
        >>> from optiland.materials import IdealMaterial, Material
        >>> from optiland.thin_film import ThinFilmStack
        >>> air, glass = IdealMaterial(1.0), IdealMaterial(1.52)
        >>> tf = ThinFilmStack(incident_material=air, substrate_material=glass)
        >>> # 100 nm SiO2 on glass
        >>> SiO2 = Material("SiO2", reference="Gao")
        >>> tf.add_layer_nm(SiO2, 100.0)
        >>> R = tf.reflectance_nm_deg([550.0], [0.0], polarization="s")
        >>> T = tf.transmittance_nm_deg([550.0], [0.0], polarization="s")
        >>> A = tf.absorptance_nm_deg([550.0], [0.0], polarization="s")
    """

    incident_material: BaseMaterial
    substrate_material: BaseMaterial
    layers: list[Layer] = field(default_factory=list)
    reference_wl_um: float | None = None
    reference_AOI_deg: float | None = 0

    def __str__(self):
        """Return a concise summary of the stack structure."""
        inc_name = getattr(
            self.incident_material, "name", self.incident_material.__class__.__name__
        )
        sub_name = getattr(
            self.substrate_material, "name", self.substrate_material.__class__.__name__
        )

        if not self.layers:
            return (
                f"ThinFilmStack(incident={inc_name}, substrate={sub_name}, layers=[])"
            )

        layer_lines: list[str] = []
        for i, layer in enumerate(self.layers, start=1):
            material_name = getattr(
                layer.material, "name", layer.material.__class__.__name__
            )
            layer_lines.append(
                f"  {i}. {material_name} ({layer.thickness_um * 1000:.1f} nm)"
            )
        layers_str = "\n".join(layer_lines)
        total_th = sum(layer.thickness_um for layer in self.layers)
        return (
            f"ThinFilmStack Summary\n"
            f"---------------------\n"
            f"Incident:  {inc_name}\n"
            f"Substrate: {sub_name}\n"
            f"Layers:\n{layers_str}\n"
            f"---------------------\n"
            f"Total Thickness: {total_th * 1000:.1f} nm"
        )

    def copy(
        self,
        incident: BaseMaterial | None = None,
        substrate: BaseMaterial | None = None,
    ):
        """Creates a copy of the stack with optionally new surrounding materials."""
        return ThinFilmStack(
            incident_material=incident if incident else self.incident_material,
            substrate_material=substrate if substrate else self.substrate_material,
            layers=self.layers.copy(),
            reference_wl_um=self.reference_wl_um,
            reference_AOI_deg=self.reference_AOI_deg,
        )

    # ----- structure helpers -----
    def add_layer(
        self, material: BaseMaterial, thickness_um: float, name: str | None = None
    ) -> ThinFilmStack:
        """Append a layer to the stack.

        Args:
            material: Optiland material providing n(λ), k(λ).
            thickness_um: Thickness in microns (µm).
            name: Optional label.

        Returns:
            self for chaining.
        """
        self.layers.append(Layer(material, thickness_um, name))
        return self

    def add_layer_nm(
        self, material: BaseMaterial, thickness_nm: float, name: str | None = None
    ) -> ThinFilmStack:
        """Append a layer, thickness in nm.

        Args:
            material: Optiland material providing n(λ), k(λ).
            thickness_nm: Thickness in nanometers.
            name: Optional label.
        """
        return self.add_layer(material, thickness_nm / 1000.0, name)

    def add_layer_qwot(
        self,
        material: BaseMaterial,
        qwot_thickness: float = 1.0,
        name: str | None = None,
    ) -> ThinFilmStack:
        """Append a quarter-wave optical thickness (QWOT) layer at the reference
        wavelength and angle of incidence.

        Args:
            material: Optiland material providing n(λ), k(λ).
            name: Optional label.

        Raises:
            ValueError: If reference_wl_um is not set.
        """
        if self.reference_wl_um is None:
            raise ValueError("reference_wl_um must be set for adding QWOT layer")
        wl_um = self.reference_wl_um
        th_rad = 0.0
        if self.reference_AOI_deg is not None:
            th_rad = be.deg2rad(self.reference_AOI_deg)
        n = float(be.atleast_1d(material.n(wl_um))[0])  # to ensure scalar float
        thickness_um = qwot_thickness * wl_um / (4 * n * be.cos(th_rad))
        return self.add_layer(thickness_um=thickness_um, material=material, name=name)

    # ----- units helpers -----
    @staticmethod
    def _to_um(wavelength_um_or_nm: float | Array, assume_nm: bool = False):
        arr = be.atleast_1d(wavelength_um_or_nm)
        return arr / 1000.0 if assume_nm else arr

    @staticmethod
    def _deg_to_rad(angle_deg: float | Array):
        return be.atleast_1d(angle_deg) * (be.pi / 180.0)

    # ----- public API: coefficients -----
    def compute_rtRTA(
        self,
        wavelength_um: float | Array,
        aoi_rad: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> dict[str, Any]:
        """Compute complex and power coefficients over λ×θ grids.

        Args:
            wavelength_um: Wavelength(s) in microns (scalar or array). Use helpers
            for nm.
            aoi_rad: Angle(s) of incidence in radians (scalar or array). Use helpers
            for degrees.
            polarization: 's', 'p' or 'u' (unpolarized averages powers of s and p).
                default 'u'.

        Returns:
            Dict with keys 'r','t','R','T','A'. Shapes are (Nλ, Nθ).

        Note:
        - For unpolarized 'u', r, t are s-polarization amplitudes; R, T, A are
        averaged powers.
        """
        wl = be.atleast_1d(wavelength_um)
        th = be.atleast_1d(aoi_rad)
        if polarization in ("s", "p"):
            r, t, R, T, A = _tmm_coh(self, wl[:, None], th[None, :], polarization)

            return {"r": r, "t": t, "R": R, "T": T, "A": A}
        elif polarization == "u":
            rs, ts, Rs, Ts, As = _tmm_coh(self, wl[:, None], th[None, :], "s")
            rp, tp, Rp, Tp, Ap = _tmm_coh(self, wl[:, None], th[None, :], "p")
            R = 0.5 * (Rs + Rp)
            T = 0.5 * (Ts + Tp)
            A = 0.5 * (As + Ap)
            # Return s-amplitudes for reference; intensities are averaged
            return {"r": rs, "t": ts, "R": R, "T": T, "A": A}
        else:
            raise ValueError("polarization must be 's', 'p' or 'u'")

    def compute_rtRTA_elementwise(
        self,
        wavelength_um: float | Array,
        aoi_rad: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> dict[str, Any]:
        """Compute complex and power coefficients element-wise (no grid).

        Use this when wavelength and aoi have matching shapes (e.g. per-ray).
        """
        wl = be.atleast_1d(wavelength_um)
        th = be.atleast_1d(aoi_rad)
        if polarization in ("s", "p"):
            r, t, R, T, A = _tmm_coh(self, wl, th, polarization)
            return {"r": r, "t": t, "R": R, "T": T, "A": A}
        elif polarization == "u":
            rs, ts, Rs, Ts, As = _tmm_coh(self, wl, th, "s")
            rp, tp, Rp, Tp, Ap = _tmm_coh(self, wl, th, "p")
            R = 0.5 * (Rs + Rp)
            T = 0.5 * (Ts + Tp)
            A = 0.5 * (As + Ap)
            return {"r": rs, "t": ts, "R": R, "T": T, "A": A}
        else:
            raise ValueError("polarization must be 's', 'p' or 'u'")

    def compute_rtRAT_nm_deg(
        self,
        wavelength_nm: float | Array,
        aoi_deg: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> dict[str, float | Array]:
        """Same as coefficients() but inputs in nm and degrees."""
        wl_um = self._to_um(wavelength_nm, assume_nm=True)
        th_rad = self._deg_to_rad(aoi_deg)
        return self.compute_rtRTA(wl_um, th_rad, polarization)

    # ----- convenience getters -----
    def reflectance(
        self,
        wavelength_um: float | Array,
        aoi_rad: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> Array:
        return self.compute_rtRTA(wavelength_um, aoi_rad, polarization)["R"]

    def transmittance(
        self,
        wavelength_um: float | Array,
        aoi_rad: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> Array:
        return self.compute_rtRTA(wavelength_um, aoi_rad, polarization)["T"]

    def absorptance(
        self,
        wavelength_um: float | Array,
        aoi_rad: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> Array:
        return self.compute_rtRTA(wavelength_um, aoi_rad, polarization)["A"]

    def reflectance_nm_deg(
        self,
        wavelength_nm: float | Array,
        aoi_deg: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> Array:
        return self.compute_rtRAT_nm_deg(wavelength_nm, aoi_deg, polarization)["R"]

    def transmittance_nm_deg(
        self,
        wavelength_nm: float | Array,
        aoi_deg: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> Array:
        return self.compute_rtRAT_nm_deg(wavelength_nm, aoi_deg, polarization)["T"]

    def absorptance_nm_deg(
        self,
        wavelength_nm: float | Array,
        aoi_deg: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> Array:
        return self.compute_rtRAT_nm_deg(wavelength_nm, aoi_deg, polarization)["A"]

    def RTA(
        self,
        wavelength_um: float | Array,
        aoi_rad: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> tuple[Array, Array, Array]:
        """Return (R, T, A) for given wavelength(s) in µm and AOI(s) in radians."""
        rta_data = self.compute_rtRTA(wavelength_um, aoi_rad, polarization)
        return (
            rta_data["R"],
            rta_data["T"],
            rta_data["A"],
        )

    def RTA_nm_deg(
        self,
        wavelength_nm: float | Array,
        aoi_deg: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> tuple[Array, Array, Array]:
        """Return (R, T, A) for given wavelength(s) in nm and AOI(s) in degrees."""
        rta_data = self.compute_rtRAT_nm_deg(wavelength_nm, aoi_deg, polarization)
        return (
            rta_data["R"],
            rta_data["T"],
            rta_data["A"],
        )

    # ----- insertion / removal helpers -----
    def insert_layer(
        self,
        index: int,
        material: BaseMaterial,
        thickness_um: float,
        name: str | None = None,
    ) -> ThinFilmStack:
        """Insert a layer at an arbitrary position.

        Args:
            index: Position to insert at (0 = closest to incident).
            material: Optiland material providing n(λ), k(λ).
            thickness_um: Thickness in microns (µm).
            name: Optional label.

        Returns:
            self for chaining.
        """
        self.layers.insert(index, Layer(material, thickness_um, name))
        return self

    def insert_layer_nm(
        self,
        index: int,
        material: BaseMaterial,
        thickness_nm: float,
        name: str | None = None,
    ) -> ThinFilmStack:
        """Insert a layer at an arbitrary position, thickness in nm.

        Args:
            index: Position to insert at (0 = closest to incident).
            material: Optiland material providing n(λ), k(λ).
            thickness_nm: Thickness in nanometers.
            name: Optional label.

        Returns:
            self for chaining.
        """
        return self.insert_layer(index, material, thickness_nm / 1000.0, name)

    def remove_layer(self, index: int) -> Layer:
        """Remove and return the layer at *index*.

        Args:
            index: Index of the layer to remove.

        Returns:
            The removed Layer.
        """
        return self.layers.pop(index)

    def split_layer(self, layer_index: int, position_fraction: float) -> ThinFilmStack:
        """Split a layer into two layers of the same material.

        The original layer at *layer_index* is replaced by two layers whose
        combined thickness equals the original.  Useful for needle insertion
        *within* a layer.

        Args:
            layer_index: Index of the layer to split.
            position_fraction: Fraction (0..1) at which to split.  0.3 means
                the first sub-layer gets 30 % of the original thickness.

        Returns:
            self for chaining.
        """
        if not 0.0 < position_fraction < 1.0:
            raise ValueError("position_fraction must be strictly between 0 and 1")
        layer = self.layers[layer_index]
        t1 = layer.thickness_um * position_fraction
        t2 = layer.thickness_um * (1.0 - position_fraction)
        self.layers[layer_index] = Layer(layer.material, t1, layer.name)
        self.layers.insert(layer_index + 1, Layer(layer.material, t2, layer.name))
        return self

    def deep_copy(self) -> ThinFilmStack:
        """Create a deep copy with new Layer instances (materials are shared).

        Returns:
            A new ThinFilmStack with independent layers.
        """
        new_layers = [
            Layer(layer.material, layer.thickness_um, layer.name)
            for layer in self.layers
        ]
        return ThinFilmStack(
            incident_material=self.incident_material,
            substrate_material=self.substrate_material,
            layers=new_layers,
            reference_wl_um=self.reference_wl_um,
            reference_AOI_deg=self.reference_AOI_deg,
        )

    def __len__(self):
        return len(self.layers)

    def __repr__(self):
        parts = [layer.name or f"Layer({i})" for i, layer in enumerate(self.layers)]
        return f"ThinFilmStack({len(self.layers)} layers: " + " -> ".join(parts) + ")"

    def _stack_material_colors(self) -> dict[str, str]:
        """Assign a unique color to each distinct material in the stack."""
        import matplotlib.colors as mcolors

        color_cycle = list(mcolors.TABLEAU_COLORS.values())
        material_names = (
            [_material_display_name(self.incident_material)]
            + [_material_display_name(layer.material) for layer in self.layers]
            + [_material_display_name(self.substrate_material)]
        )
        return {
            name: color_cycle[i % len(color_cycle)]
            for i, name in enumerate(dict.fromkeys(material_names))
        }

    def _compute_stack_bands(self) -> list[_StackBand]:
        """Compute the substrate/layer/incident-medium bands for plot_structure.

        Layout only — no matplotlib calls — so it can be reused by both the
        rendering code and tests.
        """
        colors = self._stack_material_colors()

        total_layer_thickness = _to_float(
            sum(layer.thickness_um for layer in self.layers)
        )
        # Ensure minimum thickness for visualization (avoid singular ylim
        # on empty stacks)
        if total_layer_thickness == 0:
            total_layer_thickness = 1.0

        incident_thickness = 0.08 * total_layer_thickness
        substrate_thickness = 0.08 * total_layer_thickness

        bands = []

        # Substrate (bottom, negative y)
        substrate_name = _material_display_name(self.substrate_material)
        bands.append(
            _StackBand(
                y=-substrate_thickness,
                height=substrate_thickness,
                color=colors[substrate_name],
                label=substrate_name,
                text=substrate_name,
            )
        )

        # Layers (middle, positive y)
        y = 0.0
        for layer in self.layers:
            material_name = _material_display_name(layer.material)
            label = layer.name or material_name
            if label:
                label = re.sub(r"\d+", lambda m: str(int(m.group())), label)
            height = _to_float(layer.thickness_um)
            bands.append(
                _StackBand(
                    y=y,
                    height=height,
                    color=colors[material_name],
                    label=label,
                )
            )
            y += height

        # Incident medium (top)
        incident_name = _material_display_name(self.incident_material)
        bands.append(
            _StackBand(
                y=y,
                height=incident_thickness,
                color=colors[incident_name],
                label=incident_name,
                text=incident_name,
            )
        )

        return bands

    def plot_structure(self, ax: plt.Axes = None) -> tuple[plt.Figure, plt.Axes]:
        """Plots a schematic representation of the thin film stack structure.
        This method visualizes the stack as a series of colored rectangles, each
        representing a material layer, the substrate, and the incident medium.
        Each rectangle's height corresponds to the physical thickness of the
        layer (in micrometers), and colors are assigned uniquely to each material.
        The substrate is plotted at the bottom, followed by the stack layers,
        and the incident medium at the top. Material names or refractive indices
        are used as labels in the legend.

        Args:
            ax (plt.Axes, optional): The axes on which to plot the structure.
                If None, a new figure and axes are created.

        Returns:
            tuple[plt.Figure, plt.Axes]: The matplotlib Figure and Axes objects
                containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        bands = self._compute_stack_bands()

        for band in bands:
            ax.add_patch(
                plt.Rectangle(
                    (0, band.y),
                    1,
                    band.height,
                    color=band.color,
                    label=band.label,
                    alpha=0.7,
                )
            )
            if band.text is not None:
                ax.text(
                    0.5,
                    band.y + band.height / 2,
                    band.text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    rotation=0,
                )

        y_min = bands[0].y
        y_max = bands[-1].y + bands[-1].height

        ax.set_xlim(0, 1)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Thickness (µm)")
        ax.set_xticks([])
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles, strict=False))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            borderaxespad=0.0,
            ncol=1,
        )
        fig = ax.figure
        return fig, ax

    def plot_structure_thickness(
        self, ax: plt.Axes = None
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plots the thickness of each layer in the thin film stack as a bar chart.
        Each bar represents a layer, with its height corresponding to the layer's
        thickness in nanometers.
        Bars are colored according to the material of each layer, and a legend is
        provided to identify materials.
        Args:
            ax (plt.Axes, optional): The matplotlib Axes object to plot on.
                If None, a new figure and axes will be created.

        Returns:
            tuple[plt.Figure, plt.Axes]: The matplotlib Figure and Axes objects
                containing the plot.
        """

        if ax is None:
            fig, ax = plt.subplots()
        import matplotlib.colors as mcolors

        ax.grid(True, alpha=0.3)
        color_cycle = list(mcolors.TABLEAU_COLORS.values())

        def _to_float(value) -> float:
            if hasattr(value, "detach") and hasattr(value, "cpu"):
                value = value.detach().cpu()
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)

        def _get_name(obj):
            name = getattr(obj, "name", "") or ""
            if isinstance(obj, IdealMaterial):
                name = f"$n$ = {_to_float(obj.index[0])}"
            return name

        material_names = [_get_name(layer.material) for layer in self.layers]
        unique_materials = {
            name: color_cycle[i % len(color_cycle)]
            for i, name in enumerate(dict.fromkeys(material_names))
        }
        colors = [unique_materials[_get_name(layer.material)] for layer in self.layers]
        thicknesses_nm = [_to_float(layer.thickness_um * 1000) for layer in self.layers]
        labels = [layer.name or _get_name(layer.material) for layer in self.layers]

        indices = list(range(len(self.layers)))
        bars = ax.bar(
            indices,
            thicknesses_nm,
            color=colors,
            edgecolor=None,
            alpha=0.7,
            width=1,
        )
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Thickness (nm)")

        # Legend
        by_label = {}
        for bar, label in zip(bars, labels, strict=False):
            if label not in by_label:
                by_label[label] = bar
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            borderaxespad=0.0,
            ncol=1,
        )
        ax.set_xlim(0.5, len(self.layers) - 0.5)
        fig = ax.figure
        return fig, ax
