"""Default Merit Block

One :class:`~optiland.optimization.block.base.ResidualBlock` holding a classical
default merit: per-ray residuals on a ``"gaussian_quad"`` pupil at every selected
field and wavelength. With the quadrature weight ``q`` of each ray,

* ``kind="spot"`` gives the transverse ``dx`` and ``dy`` on the image surface, and
* ``kind="wavefront"`` gives the OPD in waves,

each times ``sqrt(q * field_weight * wavelength_weight) * weight``. So
``sum(r ** 2)`` estimates ``weight ** 2`` times the weighted sum of the squared
RMS spot sizes or wavefront errors over the fields and wavelengths.

``reference="centroid"`` measures from the quadrature-weighted centroid and
``"chief"`` from the chief ray. A spot is measured from the reference of the
primary wavelength at its field, so lateral color stays in the merit (the first
selected wavelength stands in if the primary is not selected). A wavefront is
measured at its own wavelength, as :class:`~optiland.wavefront.opd.OPD` does:
from the chief ray on a sphere centered on the chief ray's image point, or with
piston removed on a sphere centered on the centroid. Both spheres run through
the exit pupil, unlike ``CentroidStrategy``, which sizes its sphere from the
total optical path; on the Cooke triplet that shifts the merit by about 0.1%.

Experimental: part of the optimization facade, subject to change.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be
from optiland.distribution import create_distribution
from optiland.optimization.block.base import (
    BlockContext,
    BundleRequest,
    ResidualBlock,
)
from optiland.optimization.failure import ConfigurationError
from optiland.wavefront.reference_geometry import SphericalReference
from optiland.wavefront.strategy import ChiefRayStrategy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from optiland._types import BEArray
    from optiland.optic.optic import Optic

#: Accepted values of ``kind`` and ``reference``.
KINDS: tuple[str, ...] = ("spot", "wavefront")
REFERENCES: tuple[str, ...] = ("centroid", "chief")

_SPOT_ATTRS = ("x", "y")
_WAVEFRONT_ATTRS = ("x", "y", "z", "L", "M", "N", "opd")
_DIRECTIONS = ("L", "M", "N")


def _weighted_mean(values: BEArray, weights: BEArray) -> BEArray:
    """Mean of the finite ``values`` under non-negative ``weights``."""
    valid = be.isfinite(values) & (weights > 0)
    weights = be.where(valid, weights, be.zeros_like(weights))
    values = be.where(valid, values, be.zeros_like(values))
    return be.sum(values * weights) / be.sum(weights)


def _resolve_indices(selection: Any, count: int, name: str) -> list[int]:
    """Indices chosen by ``"all"`` or a sequence of integers."""
    if isinstance(selection, str):
        if selection != "all":
            raise ConfigurationError(f"{name} must be 'all' or a list of indices.")
        return list(range(count))
    indices = [int(index) for index in selection]
    if not indices or any(index < 0 or index >= count for index in indices):
        raise ConfigurationError(f"{name} {indices} out of range for {count} entries.")
    return indices


@dataclass(frozen=True)
class _Group:
    """The rays of one (field, wavelength)."""

    field: int
    wavelength: int
    request: BundleRequest


class DefaultMeritBlock(ResidualBlock):
    """Per-ray default merit on a Gaussian quadrature pupil. *Experimental.*

    See the module docstring for the residuals. The ray set is fixed when the
    block is built; field and wavelength weights are read at each evaluation.

    Args:
        optic: The optic.
        kind: ``"spot"`` or ``"wavefront"``.
        rings: Quadrature rings.
        arms: Quadrature arms.
        reference: ``"centroid"`` or ``"chief"``.
        fields: ``"all"`` or field indices.
        wavelengths: ``"all"`` or wavelength indices.
        weight: Scale of every row; must be positive.

    Raises:
        ConfigurationError: If an argument is invalid.
    """

    def __init__(
        self,
        optic: Optic,
        kind: str = "spot",
        rings: int = 3,
        arms: int = 6,
        reference: str = "centroid",
        fields: str | Sequence[int] = "all",
        wavelengths: str | Sequence[int] = "all",
        weight: float = 1.0,
    ) -> None:
        if kind not in KINDS:
            raise ConfigurationError(f"Unknown kind {kind!r}; expected {list(KINDS)}.")
        if reference not in REFERENCES:
            raise ConfigurationError(
                f"Unknown reference {reference!r}; expected {list(REFERENCES)}."
            )
        if rings < 1 or arms < 1:
            raise ConfigurationError("rings and arms must be positive.")
        if not weight > 0:
            raise ConfigurationError("weight must be positive.")
        self.optic = optic
        self.kind = kind
        self.reference = reference
        self.weight = float(weight)
        self.fields = _resolve_indices(fields, len(optic.fields.fields), "fields")
        self.wavelengths = _resolve_indices(
            wavelengths, len(optic.wavelengths.wavelengths), "wavelengths"
        )
        primary = optic.wavelengths.primary_index
        self._reference_wavelength = (
            primary if primary in self.wavelengths else self.wavelengths[0]
        )

        pupil = create_distribution("gaussian_quad")
        pupil.generate_points(rings, arms)
        self._quadrature = pupil.weights
        self._rays = be.size(pupil.x)
        self._rows_per_ray = 2 if kind == "spot" else 1
        self._chief = kind == "wavefront" or reference == "chief"
        Px, Py = pupil.x, pupil.y
        if self._chief:  # one extra ray, after the samples: it adds no row
            Px = be.concatenate([Px, be.zeros((1,))])
            Py = be.concatenate([Py, be.zeros((1,))])

        coords = optic.fields.get_field_coords()
        waves = optic.wavelengths.get_wavelengths()
        attrs = _SPOT_ATTRS if kind == "spot" else _WAVEFRONT_ATTRS
        self._groups = [
            _Group(
                field,
                wavelength,
                BundleRequest(
                    optic,
                    float(waves[wavelength]),
                    float(coords[field][0]),
                    float(coords[field][1]),
                    Px,
                    Py,
                    attrs,
                ),
            )
            for field in self.fields
            for wavelength in self.wavelengths
        ]

    def size(self) -> int:
        """Return the number of residual rows."""
        return len(self._groups) * self._rays * self._rows_per_ray

    def trace_requests(self) -> list[BundleRequest]:
        """Return one request per (field, wavelength)."""
        return [group.request for group in self._groups]

    def describe(self) -> dict:
        """Return one summary row for the info printers."""
        return {
            "Kind": self.kind,
            "Reference": self.reference,
            "Fields": len(self.fields),
            "Wavelengths": len(self.wavelengths),
            "Rays": len(self._groups) * self._rays,
            "Weight": self.weight,
        }

    def residuals(self, context: BlockContext) -> BEArray:
        """Return the weighted rows at the current design."""
        rows = self._spot(context) if self.kind == "spot" else self._wavefront(context)
        return be.concatenate(rows)

    def _scale(self, group: _Group) -> BEArray | None:
        """Row scale of one group; None when its weight is zero."""
        field_weight = self.optic.fields.fields[group.field].weight
        wavelength_weight = self.optic.wavelengths.wavelengths[group.wavelength].weight
        product = float(field_weight) * float(wavelength_weight)
        if product == 0.0:
            return None
        return be.sqrt(self._quadrature * product) * self.weight

    def _live_weights(self, index: int) -> BEArray:
        """Quadrature weights of group ``index``, 0 for rays of dropped rows."""
        count = self._rays * self._rows_per_ray
        start = index * count
        rows = self.dropped_rows
        rows = rows[(rows >= start) & (rows < start + count)]
        if not rows.size:
            return self._quadrature
        keep = np.ones(self._rays)
        keep[(rows - start) % self._rays] = 0.0
        return self._quadrature * be.array(keep)

    def _spot(self, context: BlockContext) -> list[BEArray]:
        """Rows ``dx`` then ``dy`` of each group."""
        references = {}
        for index, group in enumerate(self._groups):
            if group.wavelength != self._reference_wavelength:
                continue
            x = context.read(group.request, "x")
            y = context.read(group.request, "y")
            if self.reference == "chief":
                references[group.field] = (x[self._rays], y[self._rays])
            else:
                live = self._live_weights(index)
                samples = x[: self._rays], y[: self._rays]
                references[group.field] = (
                    _weighted_mean(samples[0], live),
                    _weighted_mean(samples[1], live),
                )

        rows = []
        for group in self._groups:
            scale = self._scale(group)
            if scale is None:
                rows.append(be.zeros((2 * self._rays,)))
                continue
            x_ref, y_ref = references[group.field]
            rows.append(
                scale * (context.read(group.request, "x")[: self._rays] - x_ref)
            )
            rows.append(
                scale * (context.read(group.request, "y")[: self._rays] - y_ref)
            )
        return rows

    def _wavefront(self, context: BlockContext) -> list[BEArray]:
        """OPD rows of each group, in waves."""
        strategy = ChiefRayStrategy(self.optic, None)
        rows = []
        for index, group in enumerate(self._groups):
            scale = self._scale(group)
            if scale is None:
                rows.append(be.zeros((self._rays,)))
                continue
            rows.append(scale * self._opd_waves(context, group, index, strategy))
        return rows

    def _opd_waves(
        self,
        context: BlockContext,
        group: _Group,
        index: int,
        strategy: ChiefRayStrategy,
    ) -> BEArray:
        """OPD of one group: see :class:`ChiefRayStrategy` for the path terms."""
        request = group.request
        data = {name: context.read(request, name) for name in _WAVEFRONT_ATTRS}
        launch = {
            name: context.read(request, name, surface=0)
            for name in (*_SPOT_ATTRS, "z", *_DIRECTIONS)
        }

        def slice_rays(part: slice) -> SimpleNamespace:
            state = SimpleNamespace(**{k: v[part] for k, v in launch.items()})
            traced = {k: v[part] for k, v in data.items()}
            return SimpleNamespace(**traced, _launch_state=state)

        rays = slice_rays(slice(0, self._rays))
        chief = slice_rays(slice(self._rays, self._rays + 1))
        live = self._live_weights(index)
        if self.reference == "chief":
            center = chief.x[0], chief.y[0], chief.z[0]
        else:
            center = tuple(_weighted_mean(v, live) for v in (rays.x, rays.y, rays.z))
        radius = be.sqrt(
            sum(
                (c - be.ravel(p)[0]) ** 2
                for c, p in zip(center, strategy.exit_pupil_point, strict=True)
            )
        )
        geometry = SphericalReference(center, radius)
        n_image = strategy.n_image
        path = rays.opd - geometry._relative_path_length(rays, n_image)
        path = strategy._restore_launch_phase(rays, path, request.wavelength, chief)
        if self.reference == "chief":
            base = chief.opd - geometry._relative_path_length(chief, n_image)
            base = base[0]
        else:
            base = _weighted_mean(path, live)
        return (base - path) / (request.wavelength * 1e-3)
