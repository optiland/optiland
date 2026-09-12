"""Presentation-only styling of retained 2D scene artists."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PySide6.QtCore import QObject, Slot
from PySide6.QtGui import QColor

import optiland.backend as be
from optiland.visualization.system.surface import Surface2D
from optiland.visualization.system.utils import transform


@dataclass
class ArtistBinding:
    """One styled artist and the live surface identities it represents."""

    artist: object
    surfaces: tuple
    body: bool
    normal: dict
    overlay: bool = False


class LayoutHighlightController(QObject):
    """Apply current state without retracing, recalculating sag or resetting axes."""

    def __init__(self, axes, canvas, state, theme):
        super().__init__(canvas)
        self.axes, self.canvas, self.state = axes, canvas, state
        self.theme = theme
        self.document = None
        self.bindings = []
        state.changed.connect(self.apply)

    def clear(self):
        for binding in self.bindings:
            if binding.overlay:
                if binding.artist.axes is not None:
                    binding.artist.remove()
            else:
                binding.artist.set(**binding.normal)
        self.bindings = []
        self.document = None

    def install(self, artists, scene_optic, surface_identities=None):
        """Bind a scene to live identities captured with its accepted request."""
        identities = tuple(surface_identities or self.state.surfaces)
        scene_surfaces = tuple(scene_optic.surfaces)
        if len(identities) != len(scene_surfaces):
            self.clear()
            return
        lookup = {
            id(scene): live
            for scene, live in zip(scene_surfaces, identities, strict=True)
        }
        represented = set()
        boundaries = {}
        bodies, surfaces = {}, {}
        for artist, component in artists.items():
            if isinstance(artist, Patch):
                owned = getattr(component, "artist_surfaces", {}).get(artist, ())
                live = tuple(
                    lookup[id(surface)] for surface in owned if id(surface) in lookup
                )
                if live:
                    bodies[artist] = live
                    represented.update(id(surface) for surface in live)
                for surface, coordinates in getattr(
                    component, "boundary_coordinates", {}
                ).items():
                    if id(surface) in lookup:
                        boundaries[lookup[id(surface)]] = coordinates
            elif isinstance(artist, Line2D):
                # Raw Surface values in the public map identify aperture
                # indicator chords, not the actual surface curve. Keep those
                # indicators at normal width; a selected face must be exact.
                surface = getattr(component, "surf", None)
                if id(surface) in lookup:
                    live = lookup[id(surface)]
                    surfaces[artist] = live
                    represented.add(id(live))

        # Reference planes omitted by the ordinary renderer receive a short,
        # dashed schematic surface marker only while selected/hovered. Adding
        # an artist (rather than plotting it) leaves data limits unchanged.
        span = abs(self.axes.get_ylim()[1] - self.axes.get_ylim()[0])
        radius = max(span * 0.025, 0.1)
        references = {}
        for scene, live in zip(scene_surfaces, identities, strict=True):
            if id(live) in represented or getattr(scene, "is_infinite", False):
                continue
            marker_surface = Surface2D(scene, radius)
            x, y, z = marker_surface._compute_sag("YZ")
            x, y, z = transform(x, y, z, scene, is_global=False)
            z, y = be.to_numpy(z), be.to_numpy(y)
            if not (np.isfinite(z) & np.isfinite(y)).any():
                continue
            references[live] = (z, y)
        self.install_bindings(bodies, surfaces, boundaries, references)

    def install_bindings(
        self, body_artists, surface_artists, boundary_coordinates, reference_coordinates
    ):
        """Bind already prepared geometry; arguments contain GUI-local identities.

        Workers transport indices and numeric arrays only. After accepting a
        result the GUI maps those indices through its captured surface tuple.
        No reconstructed Optic, geometry computation or tracing is required here.
        """
        self.clear()
        self.document = self.state.document
        for artist, surfaces in body_artists.items():
            self._add(artist, tuple(surfaces), body=True)
        for artist, surface in surface_artists.items():
            self._add(artist, (surface,), body=False)
        for surface, coordinates in boundary_coordinates.items():
            line = Line2D(*coordinates, visible=False, zorder=4)
            self.axes.add_artist(line)
            self._add(line, (surface,), body=False, overlay=True)
        for surface, (z, y) in reference_coordinates.items():
            line = Line2D(z, y, linestyle="--", visible=False, zorder=4)
            if np.allclose([z[0], y[0]], [z[-1], y[-1]]):
                line.set_marker("+")
            self.axes.add_artist(line)
            self._add(line, (surface,), body=False, overlay=True)
        self.apply()

    def _add(self, artist, surfaces, *, body, overlay=False):
        normal = {"linewidth": artist.get_linewidth(), "visible": artist.get_visible()}
        if body:
            normal.update(
                facecolor=artist.get_facecolor(), edgecolor=artist.get_edgecolor()
            )
        else:
            normal["color"] = artist.get_color()
        self.bindings.append(ArtistBinding(artist, surfaces, body, normal, overlay))

    @Slot()
    def apply(self):
        # Match the desktop theme's selection accent (dark) / focused
        # selection accent (light), rather than an unrelated native palette.
        selected = QColor("#007ACC" if self.theme() == "dark" else "#6C757D")
        hover = selected.lighter(140)
        for binding in self.bindings:
            kind = (
                self.state.state_for(binding.surfaces)
                if self.document is self.state.document
                else "normal"
            )
            artist = binding.artist
            if kind == "normal":
                artist.set(**binding.normal)
                continue
            color = selected if kind == "selected" else hover
            rgb = (color.redF(), color.greenF(), color.blueF())
            if binding.body:
                artist.set_facecolor((*rgb, 0.28 if kind == "selected" else 0.18))
                artist.set_edgecolor(rgb)
                artist.set_linewidth(1.5)
            else:
                artist.set_color(rgb)
                artist.set_linewidth(2.4 if kind == "selected" else 2.1)
            artist.set_visible(True)
        if self.bindings:
            self.canvas.draw_idle()
