"""Surface Group

This module contains the SurfaceGroup class, which represents a group of
surfaces in an optical system. The SurfaceGroup class provides methods for
tracing rays through the surfaces, adding and removing surfaces, and
converting the group to and from a dictionary for serialization.

Kramer Harrison, 2024
"""

from __future__ import annotations

import copy
from contextlib import suppress
from copy import deepcopy
from functools import cached_property
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.coatings import BaseCoatingPolarized
from optiland.paraxial_path import ParaxialPath, build_paraxial_path, transverse_basis
from optiland.surfaces.factories.surface_factory import SurfaceFactory
from optiland.surfaces.standard_surface import Surface

if TYPE_CHECKING:
    from optiland._types import SurfaceType
    from optiland.materials import BaseMaterial


class SurfaceGroup:
    """Represents a group of surfaces in an optical system.

    Attributes:
        surfaces (list): List of surfaces in the group.
        _last_thickness (float): The thickness of the last surface added.

    """

    def __init__(self, surfaces: list[Surface] | None = None):
        """Initializes a new instance of the SurfaceGroup class.

        Args:
            surfaces (List, optional): List of surfaces to initialize the
                group with. Defaults to None.

        """
        if surfaces is None:
            self._surfaces = []
        else:
            self._surfaces = surfaces
            self._update_surface_links()

        self.surface_factory = SurfaceFactory(self)

    def __deepcopy__(self, memo: dict) -> SurfaceGroup:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            object.__setattr__(result, k, copy.deepcopy(v, memo))
        result._rewire_observers()
        return result

    def _update_surface_links(self):
        with suppress(KeyError):
            self.__dict__.pop("surfaces")
        surfaces = self._surfaces
        if surfaces:
            surfaces[0].previous_surface = None

            if len(surfaces) > 1:
                for idx, surface in enumerate(surfaces[1:]):
                    surface.previous_surface = surfaces[idx]

    def _rewire_observers(self) -> None:
        """Re-establish material-change callbacks across the surface chain.

        Called after deepcopy so downstream surfaces are notified when an
        upstream surface's material changes.
        """
        for i in range(len(self._surfaces) - 1):
            self._surfaces[i + 1].subscribe(
                self._surfaces[i + 1]._on_upstream_material_change
            )

    def __add__(self, other: SurfaceGroup) -> SurfaceGroup:
        """Add two SurfaceGroup objects together.

        Drops the image surface of ``self`` (last element) and the object
        surface of ``other`` (first element), then stitches the remaining
        surfaces so that ``other``'s first physical surface is placed at the
        position immediately after ``self``'s last surface (accounting for
        its thickness).

        ``other`` is never mutated; a deep copy of its surfaces is taken
        before any coordinate adjustments are applied.

        Note:
            ``self``'s last surface is treated as the image-plane marker and
            is dropped from the combined group.  For correct composition,
            append a flat (radius = ∞) image surface to ``self`` before
            calling ``__add__``.  If ``self``'s last surface is a real
            optical element with finite thickness, its propagation space is
            still honoured via the junction-z calculation even though the
            surface itself is not retained.
        """
        # Deep-copy other's surfaces so the original is never mutated and the
        # combined group does not share mutable objects with other.
        other_copies = [deepcopy(s) for s in other._surfaces]

        # Junction z: the position after self's last surface (= self's image plane).
        # Using last.z + last.thickness correctly handles the case where the last
        # surface has a finite propagation space before the focal plane.
        last = self._surfaces[-1]
        last_z = float(last.geometry.cs.z)
        last_thickness = last.thickness
        if hasattr(last_thickness, "item"):
            last_thickness = last_thickness.item()

        if be.isinf(be.array(last_thickness)):
            junction_z = last_z
        else:
            junction_z = last_z + float(last_thickness)

        # Compute the global shift for other's surfaces so that other[0]
        # (the object plane) coincides with junction_z.
        object_distance = float(other_copies[0].geometry.cs.z)
        if be.isfinite(be.array(object_distance)):
            offset = junction_z - object_distance
        else:
            offset = junction_z

        # Shift other's physical surfaces (index 1 onwards) into the global frame.
        for surf in other_copies[1:]:
            surf.geometry.cs.z = be.array(float(surf.geometry.cs.z) + offset)

        # Remove stop from other if self already has one (preserving self's stop).
        # Otherwise, we keep other's stop surface as the system stop.
        self_has_stop = any(surf.is_stop for surf in self._surfaces)
        if self_has_stop:
            for surface in other_copies:
                surface.is_stop = False

        # Drop self's image surface (last) and other's object surface (first).
        return SurfaceGroup(self._surfaces[:-1] + other_copies[1:])

    @cached_property
    def surfaces(self):
        return tuple(item for item in self._surfaces)

    def __getitem__(self, index):
        return self._surfaces[index]

    def __iter__(self):
        return iter(self._surfaces)

    def __len__(self):
        return len(self._surfaces)

    def index(self, value):
        """Return the first index of the specified surface."""
        return self._surfaces.index(value)

    def clear(self):
        """Clears the list of surfaces."""
        self._surfaces = []
        self._update_surface_links()

    @property
    def x(self):
        """np.array: x intersection points on all surfaces"""
        return be.stack([surf.x for surf in self.surfaces if be.size(surf.x) > 0])

    @property
    def y(self):
        """np.array: y intersection points on all surfaces"""
        return be.stack([surf.y for surf in self.surfaces if be.size(surf.y) > 0])

    @property
    def z(self):
        """np.array: z intersection points on all surfaces"""
        return be.stack([surf.z for surf in self.surfaces if be.size(surf.z) > 0])

    @property
    def L(self):
        """np.array: x direction cosines on all surfaces"""
        return be.stack([surf.L for surf in self.surfaces if be.size(surf.L) > 0])

    @property
    def M(self):
        """np.array: y direction cosines on all surfaces"""
        return be.stack([surf.M for surf in self.surfaces if be.size(surf.M) > 0])

    @property
    def N(self):
        """np.array: z direction cosines on all surfaces"""
        return be.stack([surf.N for surf in self.surfaces if be.size(surf.N) > 0])

    @property
    def opd(self):
        """np.array: optical path difference recorded on all surfaces"""
        return be.stack([surf.opd for surf in self.surfaces if be.size(surf.opd) > 0])

    @property
    def u(self):
        """np.array: paraxial ray angles on all surfaces"""
        return be.stack([surf.u for surf in self.surfaces if be.size(surf.u) > 0])

    @property
    def intensity(self):
        """np.array: ray intensities on all surfaces"""
        return be.stack(
            [surf.intensity for surf in self.surfaces if be.size(surf.intensity) > 0]
        )

    def build_paraxial_path(self) -> ParaxialPath:
        """Build the shared folded-path metadata for the current geometry.

        The path is a per-operation snapshot -- geometry is mutable, so it
        is rebuilt rather than cached. High-level operations should build it
        once and pass it through their call chain instead of re-deriving
        frames, directions and parity in each consumer.
        """
        return build_paraxial_path(list(self.surfaces))

    @property
    def positions(self):
        """np.array: signed unfolded axial positions of surface vertices.

        The axial coordinate is the one the paraxial model is written in: a
        1-D scalar coordinate along the unfolded optical axis, with each
        reflection reversing the direction of travel (so spacings after an
        odd number of mirrors are negative). While every leg of the beam
        path runs along ±z entered along +z -- straight systems and mirrors
        at normal incidence -- that is exactly the global z of each vertex,
        and this property returns it unchanged.

        A mirror that folds the beam off the z axis (or an entry along any
        other direction, including -z) breaks that equivalence: two vertices
        on opposite sides of a 90° fold can share a z, so their spacing
        would read as zero. For those systems the coordinate is continued as
        signed cumulative vertex-to-vertex path length along the beam, which
        is what the surrounding first-order machinery -- pupil locations,
        paraxial ray heights, solves -- needs to stay correct through a
        fold.

        This is an unfolded axial scalar, never a Cartesian coordinate: use
        ``global_z_positions`` for the global z component of the vertices,
        and ``vertices_gcs`` for full three-dimensional vertex positions.
        """
        return self.build_paraxial_path().axial_positions.reshape(-1, 1)

    @property
    def global_z_positions(self):
        """np.array: global z coordinates of the surface vertices.

        Unlike :attr:`positions` this is a real-space coordinate, never an
        unfolded one. Use it for anything that has to place something in the
        global frame (drawing limits, reference spheres); use ``positions``
        for first-order calculations.
        """
        positions = be.array(
            [surf.geometry.cs.position_in_gcs[2] for surf in self.surfaces]
        )
        return positions.reshape(-1, 1)

    @property
    def vertices_gcs(self):
        """np.array: full 3-D surface vertex positions in global coordinates.

        Shape ``(num_surfaces, 3)``. Use this whenever a consumer needs a
        real-space point; ``positions`` is an unfolded axial scalar and
        ``global_z_positions`` is only the z component of these vertices.
        """
        path = self.build_paraxial_path()
        rows = [
            be.array([vertex[0], vertex[1], vertex[2]]) for vertex in path.vertices_gcs
        ]
        return be.stack(rows) if rows else be.array([])

    def _entry_frame(self, path: ParaxialPath | None = None):
        """Object-space entry frame, or ``None`` on the canonical z axis.

        ``None`` is the common case -- every leg on ±z, entered along +z
        through global x = y = 0 -- and keeps z-based callers (the paraxial
        ray aimer, the angle-field launch seeds) on their existing code path
        bit-for-bit. For any other aiming geometry (folded, off-axis entry,
        -z entry, or a system rigidly translated off the z axis) it returns
        ``(anchor, axial, direction, u, v)``: the first physical surface's
        vertex and its axial coordinate, the unit entry direction, and the
        transverse basis completing it.

        Object-space constructs live on the entry line, anchored at the
        first physical surface's vertex -- a decentered first vertex shifts
        the pupil line with it. The entrance pupil is the stop imaged into
        object space, so a launch ray aims at its apparent, unfolded
        position ``anchor + (axial_pupil - axial) * direction``, never at a
        refolded downstream point; the fold mirrors then carry the ray onto
        the physical stop. The basis gauge (and its pole near ±y entry) is
        documented on :func:`optiland.paraxial_path.transverse_basis`.

        Args:
            path: Optional prebuilt path to reuse (avoids a second walk).
        """
        if path is None:
            if len(self.surfaces) < 2:
                return None
            path = self.build_paraxial_path()
        if path.num_surfaces < 2 or path.legacy_aiming_compatible:
            return None
        return path.entry_frame()

    @staticmethod
    def _transverse_basis(direction):
        """Deterministic transverse pair ``(u, v)`` completing ``direction``.

        Delegates to :func:`optiland.paraxial_path.transverse_basis`; see
        there for the gauge convention and the pole near ±y.
        """
        return transverse_basis(direction)

    @property
    def radii(self):
        """np.array: radii of curvature of all surfaces"""
        return be.array([surf.geometry.radius for surf in self.surfaces])

    @property
    def conic(self):
        """be.array: conic constant of all surfaces"""
        values = []
        for surf in self.surfaces:
            try:
                values.append(surf.geometry.k)
            except AttributeError:
                values.append(0)
        return be.array(values)

    @property
    def stop_index(self):
        """int: the index of the aperture stop surface"""
        for index, surface in enumerate(self.surfaces):
            if surface.is_stop:
                return index

        raise ValueError(
            "No stop surface found. Exactly one surface must be marked as "
            "the aperture stop, either by passing is_stop=True to "
            "lens.add_surface(...) or by setting "
            "lens.surface_group.stop_index = <index>."
        )

    @stop_index.setter
    def stop_index(self, index: int):
        if index < 1 or index > len(self.surfaces) - 2:
            raise ValueError(
                f"Invalid stop index, got {index}. The stop must be an "
                f"optical surface, so its index must lie between 1 and "
                f"{len(self.surfaces) - 2} (index 0 is the object surface "
                f"and index {len(self.surfaces) - 1} is the image surface)."
            )
        for idx, surf in enumerate(self.surfaces):
            surf.is_stop = index == idx

    @property
    def num_surfaces(self):
        """int: the number of surfaces"""
        return len(self.surfaces)

    @property
    def uses_polarization(self):
        """bool: True if any surface uses polarization, False otherwise"""
        for surf in self.surfaces:
            if isinstance(surf.interaction_model.coating, BaseCoatingPolarized):
                return True
        return False

    @property
    def total_track(self):
        """float: the span of the unfolded signed axial surface coordinates.

        This is the extent of :attr:`positions` over the physical surfaces
        (object excluded) -- the track length of the scalar paraxial system.
        For straight systems it equals the global-z span. For a folded
        system it is the span along the unfolded axis (the length the
        equivalent unfolded system would have), which is what the
        ``total_track`` optimization operand constrains. Use
        :attr:`global_z_span` for the extent of the vertices in global z.
        """
        if self.num_surfaces < 2:
            raise ValueError(
                f"Cannot compute the total track: the system has "
                f"{self.num_surfaces} surface(s), and at least 2 are "
                "required. Add surfaces with lens.add_surface(...)."
            )
        z = self.positions[1:]
        return be.max(z) - be.min(z)

    @property
    def global_z_span(self):
        """float: the extent of the surface vertices in global z.

        The old (pre-fold-aware) meaning of ``total_track``: the span of the
        global z coordinates of the physical surface vertices. For straight
        systems this equals :attr:`total_track`; for folded systems it is
        the bounding extent in z, not a track length along the beam.
        """
        if self.num_surfaces < 2:
            raise ValueError(
                f"Cannot compute the global z span: the system has "
                f"{self.num_surfaces} surface(s), and at least 2 are "
                "required. Add surfaces with lens.add_surface(...)."
            )
        z = self.global_z_positions[1:]
        return be.max(z) - be.min(z)

    def n(self, wavelength):
        """Get the refractive indices of the surfaces.

        Args:
            wavelength (float or str, optional): The wavelength for which to
                calculate the refractive indices.

        Returns:
            numpy.ndarray: The refractive indices of the surfaces.

        """
        n = []
        for surface in self.surfaces:
            n.append(be.atleast_1d(surface.material_post.n(wavelength)))
        return be.ravel(be.array(n))

    def get_thickness(self, surface_number):
        """Calculate the thickness between two surfaces.

        Args:
            surface_number (int): The index of the first surface.

        Returns:
            float: The thickness between the two surfaces.

        """
        t = self.positions
        return t[surface_number + 1] - t[surface_number]

    def trace(self, rays, skip=0, record=True):
        """Trace the given rays through the surfaces.

        Args:
            rays (BaseRays): List of rays to be traced.
            skip (int, optional): Number of surfaces to skip before tracing.
                Defaults to 0.
            record (bool, optional): Whether to store per-surface snapshots of
                the ray state (positions, directions, intensity, OPD). The
                snapshots feed analyses and visualization but keep eight
                full-size arrays alive per surface; pass False when only the
                rays returned at the image are needed. Defaults to True.

        """
        self.reset()
        for surface in self.surfaces[skip:]:
            surface.trace(rays, record=record)
        return rays

    def add(
        self,
        new_surface=None,
        surface_type: SurfaceType = "standard",
        comment="",
        index=None,
        is_stop=False,
        material: str | BaseMaterial = "air",
        **kwargs,
    ):
        """Adds a new surface to the list of surfaces.

        Args:
            new_surface (Surface, optional): The new surface to add. If not
                provided, a new surface will be created based on the other
                arguments.
            surface_type (str, optional): The type of surface to create.
            comment (str, optional): A comment for the surface. Defaults to ''.
            index (int, optional): The index at which to insert the new
                surface. If not provided, the surface will be appended to the
                end of the list.
            is_stop (bool, optional): Indicates if the surface is the aperture.
            material (str, optional): The material of the surface.
                Default is 'air'.
            **kwargs: Additional keyword arguments for surface-specific
                parameters such as radius, conic, dx, dy, rx, ry, rz, aperture,
                bsdf, x, y, z.

        Raises:
            ValueError: If a new surface is provided and no index is given.
            IndexError: If the index is out of bounds for insertion, or negative.

        """
        if new_surface is None:
            if index is None:
                raise ValueError(
                    "No index was given for the new surface. Pass the "
                    "position it should occupy, e.g. "
                    "lens.add_surface(index=1, radius=50, thickness=5)."
                )

            new_surface = self.surface_factory.create_surface(
                surface_type,
                comment,
                index,
                is_stop,
                material,
                **kwargs,
            )

        # Used for surface positioning
        new_surface.thickness = kwargs.get("thickness", 0.0)
        self.surface_factory.material_factory.last_material = new_surface.material_post

        if index is None:
            self._surfaces.append(new_surface)
            self._update_surface_links()
            index = len(self._surfaces) - 1
        else:
            if index < 0:
                raise IndexError(
                    f"Cannot add a surface at index {index}: surface indices "
                    "are non-negative, counting from the object surface at "
                    "index 0."
                )
            if index > len(self._surfaces):
                raise IndexError(
                    f"Cannot add a surface at index {index}: the system "
                    f"currently has {len(self._surfaces)} surface(s), so the "
                    f"highest valid index is {len(self._surfaces)}. Surfaces "
                    "must be added in order, starting with the object "
                    "surface at index 0."
                )
            if index == 0 and len(self.surfaces) > 0:
                raise ValueError(
                    "Cannot add a surface at index 0: index 0 is the object "
                    "surface and it already exists. Insert at index 1 or "
                    "later, or remove the existing object surface first."
                )

            self._surfaces.insert(index, new_surface)
            self._update_surface_links()

            # Update coordinate systems if surface was inserted
            if not self.surface_factory.use_absolute_cs and index < (
                len(self._surfaces) - 1
            ):
                self._update_coordinate_systems(start_index=index)

        if new_surface.is_stop:
            for idx, surface in enumerate(self._surfaces):
                surface.is_stop = idx == index

    def remove(self, index):
        """Remove a surface from the list of surfaces.

        Cannot remove the object surface (index 0).
        If relative coordinate positioning is active (use_absolute_cs=False),
        this may trigger an update of subsequent surface positions.

        Args:
            index (int): The index of the surface to remove.

        Raises:
            ValueError: If attempting to remove the object surface (index 0).
            IndexError: If the index is out of bounds for the current list of surfaces.
        """
        if index == 0:
            raise ValueError("Cannot remove object surface (index 0).")

        if not (0 < index < len(self.surfaces)):
            raise IndexError(
                f"Index {index} is out of bounds for removing from list of "
                f"{len(self.surfaces)} surfaces."
            )

        num_surfaces_before_removal = len(self.surfaces)

        del self._surfaces[index]

        if not self.surface_factory.use_absolute_cs:
            was_not_last_surface = index < num_surfaces_before_removal - 1
            if was_not_last_surface:
                self._update_coordinate_systems(start_index=index)

        self._update_surface_links()

    def reset(self):
        """Resets all the surfaces in the collection.

        This method iterates over each surface in the collection and calls
            its `reset` method.
        """
        for surface in self.surfaces:
            surface.reset()

    def set_fresnel_coatings(self):
        """Set Fresnel coatings on all surfaces in the group."""
        for surface in self.surfaces[1:-1]:
            if surface.material_pre != surface.material_post:
                surface.set_fresnel_coating()

    def to_dict(self):
        """Convert the surface group to a dictionary.

        Returns:
            dict: The surface group as a dictionary.

        """
        return {"surfaces": [surface.to_dict() for surface in self.surfaces]}

    @classmethod
    def from_dict(cls, data):
        """Create a surface group from a dictionary.

        Args:
            data (dict): The dictionary to create the surface group from.

        Returns:
            SurfaceGroup: The surface group created from the dictionary.

        """
        return cls(
            [Surface.from_dict(surface_data) for surface_data in data["surfaces"]],
        )

    def _update_coordinate_systems(self, start_index):
        """Updates the coordinate systems of surfaces from start_index.

        This method is called when a surface is added, removed, or modified
        in a way that might affect the positions of subsequent surfaces,
        but only if absolute coordinate positioning (use_absolute_cs=True)
        is not being used by the coordinate system factory.

        It recalculates the z-coordinate of each surface based on the
        z-coordinate and 'thickness' attribute of the preceding surface.

        Args:
            start_index (int): The index of the surface from which to start
                            updating coordinate systems. The surface at
                            `start_index` itself will be updated if it's
                            not the object surface (index 0) and has a predecessor.
                            If `start_index` is 0, updates effectively begin
                            for surface 1 based on surface 0.
        """
        if not self._surfaces:
            return

        effective_start_index = max(start_index, 1)  # No update to object surface

        for i in range(effective_start_index, len(self._surfaces)):
            current_surface = self._surfaces[i]

            if i == 1:  # first surface lies at z=0.0 by definition
                new_z = 0.0
            else:
                prev_surface = self._surfaces[i - 1]
                thickness = prev_surface.thickness

                if hasattr(thickness, "item"):
                    thickness = thickness.item()

                if be.isinf(thickness):
                    raise ValueError(
                        f"Coordinate system update failed due to infinite "
                        f"thickness at surface {start_index - 1}"
                    )

                prev_z = prev_surface.geometry.cs.z
                if hasattr(prev_z, "item"):
                    prev_z = prev_z.item()
                new_z = float(prev_z) + thickness

            current_surface.geometry.cs.z = be.array(float(new_z))

    def flip(
        self,
        start_index: int = 0,
        end_index: int = 0,
    ):
        """Flips a segment of the surfaces in the group.

        The function will swap the materials on the Object and Image surface if both
        `start_index` and `end_index` are zero. Subgroups can be swapped by passing the
        index of the first surface and the index of the surface after the last surface
        of the group (standard Python slicing). Note that only "sensible" results are
        obtained when the material before and after the subgroup is the same (for
        example, air).

        Args:
            start_index (int, optional): The starting index of the segment of
                surfaces to flip. Defaults to 0 (include object surface).
            end_index (int, optional): The ending index (exclusive for positive,
                inclusive for negative slice behavior) of the segment of surfaces
                to flip. Defaults to 0 (up to, and including, the image surface).

        Raises:
            RuntimeError: If either `start_index` or `end_index` is zero, but not both.

        """
        n_surfaces_total = len(self._surfaces)

        if (start_index == 0 or end_index == 0) and not (
            start_index == 0 and end_index == 0
        ):
            raise RuntimeError(
                "Cannot flip object surface or image surface without flipping both"
            )
        flip_object_image_media = start_index == 0 and end_index == 0

        if flip_object_image_media:
            start_index = 1
            end_index = len(self.surfaces) - 1

        if start_index < 0:
            start_index = n_surfaces_total + start_index

        if end_index < 0:
            actual_slice_end_index = n_surfaces_total + end_index
        else:
            actual_slice_end_index = end_index

        if start_index >= actual_slice_end_index:
            # No surfaces to flip or invalid range
            self.reset()
            return

        original_indices_in_segment = list(range(start_index, actual_slice_end_index))

        if not original_indices_in_segment:
            self.reset()
            return

        # Extract the segment, reverse it, and place it back
        segment_to_reverse = self._surfaces[start_index:actual_slice_end_index]
        z_positions = be.ravel(
            be.array([surf.geometry.cs.z for surf in segment_to_reverse])
        )
        segment_to_reverse.reverse()
        self._surfaces[start_index:actual_slice_end_index] = segment_to_reverse

        # Ignore thickness attribute, determine new thickness based on z-coordinate of
        # surfaces.
        new_thickness = be.flip(
            be.diff(z_positions, prepend=be.array([z_positions[0]]))
        )
        new_thickness[-1] = (
            self._surfaces[actual_slice_end_index].geometry.cs.z - z_positions[-1]
        )
        new_z = (
            be.flip(be.diff(z_positions, append=be.array([z_positions[-1]]))).cumsum(0)
            + z_positions[0]
        )

        for surf, thickness, z in zip(
            segment_to_reverse, new_thickness, new_z, strict=True
        ):
            surf.flip()
            surf.geometry.cs.z = z
            surf.thickness = thickness

        # Special handling: flip materials on object and image surfaces if flip() called
        # without arguments
        if flip_object_image_media:
            self.surfaces[0].material_post, self.surfaces[-1].material_post = (
                self.surfaces[-1].material_post,
                self.surfaces[0].material_post,
            )
        self._update_surface_links()
        self.reset()
