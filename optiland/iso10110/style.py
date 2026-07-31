"""ISO 10110 Drawing Style

Exposes the font sizes and border margin used by both renderers
(:class:`~optiland.iso10110._mpl_renderer._MatplotlibRenderer` and
:class:`~optiland.iso10110._dxf_renderer._DxfRenderer`) as a single, user-facing
:class:`DrawingStyle` object, instead of leaving them hardcoded in the
renderer source.

Design
------
Each renderer keeps its own already-tuned base value per text role (e.g.
matplotlib draws the title-block "value" text at 7.5 pt, DXF at 2.5 mm — the
two are not on a common scale, since matplotlib sizes are typeset in points
and DXF text height is a literal drawing dimension in mm). Rather than invent
a fictitious pt<->mm conversion, each :class:`DrawingStyle` field is a
*multiplicative scale factor*, applied on top of each renderer's own base
value for that role. A factor of ``1.0`` (the default for every field except
``border_margin``) therefore reproduces the built-in appearance exactly in
both output formats; overriding one field scales that role uniformly in
whichever renderer is used.

``border_margin`` is the one absolute (non-scale) field: the sheet border
margin in mm, identical in both renderers.

Bernhard Lutzer, 2026
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

#: Names of every multiplicative scale field on DrawingStyle (everything
#: except border_margin, which is an absolute mm value).
_SCALE_FIELDS = (
    "axis_label_scale",
    "surface_label_scale",
    "symbol_annotation_scale",
    "dimension_scale",
    "component_dimension_scale",
    "table_header_scale",
    "table_body_scale",
    "lambda_symbol_scale",
    "reference_note_scale",
    "efl_annotation_scale",
    "title_label_scale",
    "title_value_scale",
    "title_notes_scale",
    "title_general_tol_scale",
)


class DrawingStyle(BaseModel):
    """Font-size scale factors and border margin for ISO 10110 drawings.

    All scale factors default to ``1.0``, reproducing the built-in
    appearance exactly. Pass a :class:`DrawingStyle` instance to
    :class:`~optiland.iso10110.drawing.ElementDrawing` or
    :class:`~optiland.iso10110.report.ISO10110Report` to override any subset
    of fields; unspecified fields keep their default of ``1.0``.

    Example::

        # Make table text and dimension callouts 20% larger everywhere,
        # and shrink the sheet border margin.
        style = DrawingStyle(table_body_scale=1.2, dimension_scale=1.2,
                             border_margin=8.0)
        report = ISO10110Report(spec, style=style)

    Fields (each multiplies both renderers' own base size for that role):

    - *axis_label_scale*: "opt. axis" / "rot. axis" labels and the
      effective-aperture (Øe) bracket label.
    - *surface_label_scale*: S1/S2/… surface index labels.
    - *symbol_annotation_scale*: the sharp-edge "0" symbol
      (ISO 10110-1 §5.9.5.2).
    - *dimension_scale*: the primary cross-section dimension callouts —
      centre thickness (CT), physical diameter (Ø), and edge thickness (ET).
    - *component_dimension_scale*: per-component thickness labels
      (``t1 = …``, ``t2 = …``) on cemented lenses.
    - *table_header_scale*: spec-table column headers (e.g.
      "SURFACE 1 (FRONT)", "MATERIAL").
    - *table_body_scale*: spec-table row text (ISO codes, nd/νd) and the
      7/ coating callout text.
    - *lambda_symbol_scale*: the encircled-λ glyph next to a coating callout.
    - *reference_note_scale*: the bottom-left "per ISO 10110; λ = …" note.
    - *efl_annotation_scale*: the "f′ = … mm" annotation.
    - *title_label_scale*: small field labels in the title block
      ("DRAWN BY", "APPROVED", "NOTES", "DIM. IN mm", …).
    - *title_value_scale*: the filled-in title-block field values (project
      name, drawn-by, date, scale, sheet, revision, …).
    - *title_notes_scale*: the free-text notes value in the title block.
    - *title_general_tol_scale*: the ISO 10110-11 general-tolerance note.
    - *border_margin*: sheet border margin in mm (absolute, not a scale
      factor). Default ``10.0`` matches the built-in layout.
    """

    model_config = ConfigDict(extra="forbid")

    axis_label_scale: float = 1.0
    surface_label_scale: float = 1.0
    symbol_annotation_scale: float = 1.0
    dimension_scale: float = 1.0
    component_dimension_scale: float = 1.0
    table_header_scale: float = 1.0
    table_body_scale: float = 1.0
    lambda_symbol_scale: float = 1.0
    reference_note_scale: float = 1.0
    efl_annotation_scale: float = 1.0
    title_label_scale: float = 1.0
    title_value_scale: float = 1.0
    title_notes_scale: float = 1.0
    title_general_tol_scale: float = 1.0

    border_margin: float = 10.0

    @model_validator(mode="after")
    def _validate(self) -> DrawingStyle:
        for name in _SCALE_FIELDS:
            v = getattr(self, name)
            if v <= 0:
                raise ValueError(f"{name} must be > 0, got {v}")
        if self.border_margin <= 0:
            raise ValueError(f"border_margin must be > 0, got {self.border_margin}")
        return self
