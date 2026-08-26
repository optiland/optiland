"""ISO 10110 Drawing Specification Overlay

DrawingSpec is the user-facing bridge between an optiland Optic (which holds
optical geometry) and the ISO 10110 drawing generator (which needs fabrication
tolerances).  It stores per-surface and per-element annotations that the optic
itself does not carry.

Typical workflow
----------------
    spec = DrawingSpec(lens)

    # Surface-level tolerances (designer fills these in)
    spec.set_surface(1, irregularity=0.5, centration=0.5, imperfections="2x0.04")
    spec.set_surface(2, irregularity=1.0, imperfections="1x0.063")

    # Element-level glass quality (optician / procurement fills these in)
    spec.set_element(0, bubbles="1x0.16", homogeneity_grade=2, striae_grade="A",
                     birefringence=10, part_number="OPT-001")

    spec.save_yaml("lens_spec.yaml")
    # later:
    spec = DrawingSpec.load_yaml("lens_spec.yaml", lens)

Bernhard Lutzer, 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from optiland.iso10110.elements import LensElement, identify_elements
from optiland.iso10110.notation import ElementSpec, SurfaceSpec

if TYPE_CHECKING:
    from optiland.optic.optic import Optic


# Title-block-only subset of ElementSpec fields (ISO 7200), settable via
# DrawingSpec.set_title() independently of the ISO 10110 optical-quality
# fields settable via set_element()/set_material().
_TITLE_BLOCK_FIELDS = frozenset(
    {"part_number", "revision", "drawn_by", "approved_by", "notes"}
)


class DrawingSpec:
    """Fabrication annotation overlay for an optic.

    Args:
        optic: The optical system to annotate.
        project_name: Optional project name printed in every title block.
    """

    def __init__(
        self,
        optic: Optic,
        project_name: str = "",
        organisation: str = "",
        reference_wavelength: float = 546.07,
    ) -> None:
        self.optic = optic
        self.project_name = project_name
        self.organisation = organisation
        self.reference_wavelength = float(reference_wavelength)
        self._elements: list[LensElement] = identify_elements(optic)
        self._surface_specs: dict[int, SurfaceSpec] = {}
        self._element_specs: dict[int, ElementSpec] = {}
        # Per-component material specs for cemented lenses.
        # Key: (element_index, component_index).  Used by drawing.py to render
        # separate 0/, 1/, 2/ rows in each glass column of a cemented doublet.
        self._material_specs: dict[tuple[int, int], ElementSpec] = {}

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_surface(self, surface_index: int, **kwargs) -> None:
        """Set ISO 10110 surface-level specs for *surface_index*.

        Keyword arguments correspond to any
        :class:`~optiland.iso10110.notation.SurfaceSpec` field. Common examples:

        - ``sag_tolerance``, ``irregularity``, ``rotationally_symmetric``,
          ``test_wavelength``, ``form_unit`` — 3/ surface form (ISO 10110-5)
        - ``centration``, ``centration_decentration``, ``pyramid_error`` — 4/
        - ``imperfections``, ``coating_imperfections``, ``scratches``,
          ``edge_chips`` — 5/ (ISO 10110-7)
        - ``laser_damage`` — 6/
        - ``coating`` — 7/ (ISO 10110-9)
        - ``roughness`` — 8/ (ISO 10110-8)
        - ``r_tolerance``, ``ca_diameter``, ``ca_tolerance`` — dimensions
        - ``chamfer``, ``chamfer_angle``, ``sharp_edge`` — edge treatment
        - ``wavefront_deformation`` — 13/ (ISO 10110-14)
        - ``assembly_imperfections`` — 15/ (ISO 10110-7 §5.5)
        - ``grating_type`` — ISO 10110-16 type symbol (for grating surfaces)

        Raises:
            pydantic.ValidationError: If an unknown keyword is given (typo'd
                or renamed field name) or a value fails validation — subclasses
                ``ValueError``. See :class:`~optiland.iso10110.notation.SurfaceSpec`.
        """
        self._surface_specs[surface_index] = SurfaceSpec(**kwargs)

    def set_element(self, element_index: int, **kwargs) -> None:
        """Set ISO 10110 element-level specs for *element_index*.

        Keyword arguments correspond to any
        :class:`~optiland.iso10110.notation.ElementSpec` field. Common examples:

        - ``birefringence`` — 0/ stress birefringence in nm/cm (ISO 10110-2)
        - ``bubbles`` — 1/ bubble-and-inclusion grade, e.g. ``"1×0.16"``
          (ISO 10110-3)
        - ``nh_class`` — 2/ homogeneity, e.g. ``"NH040"`` (ISO 10110-18)
        - ``homogeneity_grade`` — legacy 0–5 integer grade (ISO 10110-4)
        - ``striae_density`` — striae density class 1–5 (ISO 10110-4)
        - ``striae_shadowgraph`` — striae shadowgraph class A–D
        - ``striae_grade`` — legacy striae class string (backward compat)
        - ``diameter``, ``diameter_tolerance``, ``ct_tolerance`` — dimensions
        - ``matched_pair`` — True adds "M" marker to CT dimension
        - ``part_number``, ``revision``, ``drawn_by``, ``approved_by``,
          ``notes`` — title block fields (see also :meth:`set_title`)

        For cemented lenses where each glass component has its own quality
        requirements, use :meth:`set_material` to set per-component specs.

        Note this call *replaces* the entire element spec. If you also use
        :meth:`set_title`, call ``set_element`` first, since a later
        ``set_element`` call overwrites title-block fields set by an earlier
        ``set_title`` call for the same *element_index*.

        Raises:
            pydantic.ValidationError: If an unknown keyword is given (typo'd
                or renamed field name) or a value fails validation — subclasses
                ``ValueError``. See :class:`~optiland.iso10110.notation.ElementSpec`.
        """
        self._element_specs[element_index] = ElementSpec(**kwargs)

    def set_title(self, element_index: int, **kwargs) -> None:
        """Set ISO 7200 title-block fields for *element_index*.

        Keeps title-block metadata (who drew/approved a part, part number,
        revision, free-text notes) separate at the call site from the ISO
        10110 optical-quality specs set via :meth:`set_element`, per the
        API structure requested in
        `#458 <https://github.com/optiland/optiland/issues/458>`_.

        Keyword arguments (all optional):

        - ``part_number`` — drawing/part identifier
        - ``revision`` — revision letter, defaults to ``"A"``
        - ``drawn_by`` — drafter initials/name
        - ``approved_by`` — approver initials/name
        - ``notes`` — free-text note shown in the title block

        This call is additive: it merges into any spec already set by
        :meth:`set_element` for the same *element_index*, rather than
        replacing it. Call :meth:`set_element` first if you use both, since
        ``set_element`` fully replaces the stored spec and would discard
        title-block fields set by an earlier ``set_title`` call.
        """
        valid = _TITLE_BLOCK_FIELDS
        unknown = set(kwargs) - valid
        if unknown:
            raise ValueError(
                f"set_title() got unexpected field(s) {sorted(unknown)}; "
                f"valid fields: {sorted(valid)}"
            )
        existing = self._element_specs.get(element_index)
        merged = existing.to_dict() if existing is not None else {}
        merged.update(kwargs)
        self._element_specs[element_index] = ElementSpec(**merged)

    def set_material(
        self,
        element_index: int,
        component_index: int,
        **kwargs,
    ) -> None:
        """Set ISO 10110 material quality specs for one glass component.

        Use this to specify independent 0/ (birefringence), 1/ (bubbles), and
        2/ (homogeneity/striae) grades for each glass in a cemented lens.

        For a singlet, *component_index* is always ``0`` and this is equivalent
        to calling :meth:`set_element`.  For a cemented doublet, use
        ``component_index=0`` for the crown and ``component_index=1`` for the
        flint.

        The per-component spec takes precedence over the element-level spec set
        by :meth:`set_element` when the drawing is rendered.

        Keyword arguments correspond to :class:`~optiland.iso10110.notation.ElementSpec`
        fields: ``birefringence``, ``bubbles``, ``homogeneity_grade``,
        ``nh_class``, ``striae_grade``, ``striae_density``,
        ``striae_shadowgraph``.  Title-block fields (``part_number``,
        ``revision``, etc.) are only used from the element-level spec.

        Raises:
            pydantic.ValidationError: If an unknown keyword is given (typo'd
                or renamed field name) or a value fails validation — subclasses
                ``ValueError``. See :class:`~optiland.iso10110.notation.ElementSpec`.
        """
        self._material_specs[(element_index, component_index)] = ElementSpec(**kwargs)

    # ------------------------------------------------------------------
    # Public getters
    # ------------------------------------------------------------------

    @property
    def elements(self) -> list[LensElement]:
        """Inferred lens elements in optical order."""
        return self._elements

    def get_surface_spec(self, surface_index: int) -> SurfaceSpec:
        """Return the :class:`SurfaceSpec` for *surface_index*, or a blank one."""
        return self._surface_specs.get(surface_index, SurfaceSpec())

    def get_element_spec(self, element_index: int) -> ElementSpec:
        """Return the :class:`ElementSpec` for *element_index*, or a blank one."""
        return self._element_specs.get(element_index, ElementSpec())

    def get_material_spec(
        self,
        element_index: int,
        component_index: int,
    ) -> ElementSpec:
        """Return the :class:`ElementSpec` for one glass component.

        Looks up the per-component spec set by :meth:`set_material`.  Falls
        back to the element-level spec (:meth:`get_element_spec`) when no
        per-component spec has been set, so singlet drawings continue to work
        without any API change.

        Args:
            element_index: 0-based index of the lens element.
            component_index: 0-based index of the glass component within the
                element (0 for the first glass, 1 for the second, …).
        """
        key = (element_index, component_index)
        if key in self._material_specs:
            return self._material_specs[key]
        return self.get_element_spec(element_index)

    # ------------------------------------------------------------------
    # YAML persistence
    # ------------------------------------------------------------------

    def save_yaml(self, path: str | Path) -> None:
        """Serialise spec annotations to a YAML file.

        The file can be loaded back with :meth:`load_yaml` and handed to the
        same (or a reconstructed) optic.

        Args:
            path: Destination file path.
        """
        data: dict = {
            "project_name": self.project_name,
            "organisation": self.organisation,
            "reference_wavelength": self.reference_wavelength,
            "surface_specs": {
                str(k): v.to_dict() for k, v in self._surface_specs.items()
            },
            "element_specs": {
                str(k): v.to_dict() for k, v in self._element_specs.items()
            },
            "material_specs": {
                f"{k[0]},{k[1]}": v.to_dict() for k, v in self._material_specs.items()
            },
        }
        Path(path).write_text(
            yaml.dump(data, default_flow_style=False), encoding="utf-8"
        )

    @classmethod
    def load_yaml(cls, path: str | Path, optic: Optic) -> DrawingSpec:
        """Load a previously saved spec from *path* and attach to *optic*.

        Args:
            path: YAML file written by :meth:`save_yaml`.
            optic: The optic the spec should annotate.

        Returns:
            A populated :class:`DrawingSpec` instance.
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        spec = cls(
            optic,
            project_name=raw.get("project_name", ""),
            organisation=raw.get("organisation", ""),
            reference_wavelength=raw.get("reference_wavelength", 546.07),
        )

        for k, d in raw.get("surface_specs", {}).items():
            spec._surface_specs[int(k)] = SurfaceSpec.from_dict(d)

        for k, d in raw.get("element_specs", {}).items():
            spec._element_specs[int(k)] = ElementSpec.from_dict(d)

        for k, d in raw.get("material_specs", {}).items():
            elem_idx, comp_idx = map(int, k.split(","))
            spec._material_specs[(elem_idx, comp_idx)] = ElementSpec.from_dict(d)

        return spec

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        n_elem = len(self._elements)
        n_ssurf = len(self._surface_specs)
        n_espec = len(self._element_specs)
        return (
            f"DrawingSpec("
            f"{n_elem} elements, "
            f"{n_ssurf} surface specs, "
            f"{n_espec} element specs)"
        )
