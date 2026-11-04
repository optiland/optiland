"""CODE V File Exporter

Entry point for exporting an Optiland Optic to a CODE V Sequential .seq file.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.fileio.base import BaseOpticWriter
from optiland.fileio.codev.writer.encoder import CodeVFileEncoder
from optiland.fileio.codev.writer.formatter import OpticToCodeVConverter

if TYPE_CHECKING:
    from optiland.optic import Optic

__all__ = ["save_codev_file", "CodeVWriter"]


def save_codev_file(optic: Optic, filepath: str) -> None:
    """Export an Optic to a CODE V Sequential .seq file.

    The file is written in UTF-8 encoding. Everything that cannot be written
    faithfully is refused before the destination file is opened, so an existing
    file is left untouched. Features that are dropped or changed are reported
    with a ``UserWarning``.

    The optic name becomes the ``TITLE``; a ``'`` in it is written as ``"`` and a
    line break as a space. ``STO`` is written once, on the stop surface Optiland
    reports (``optic.surfaces.stop_index``).

    Args:
        optic: The optic to export.
        filepath: Destination path (should end in ``.seq``).

    Raises:
        NotImplementedError: If the optic has a material with a custom
            propagation model, a non-air medium in object space, a curved
            object surface, the aperture
            stop on the object or image surface, a negative first spacing
            before any mirror (a system authored backwards, with the light
            starting out towards ``-z``; a negative object distance alone is
            written), a
            decentered, tilted or chained surface, a surface geometry other
            than planar, spherical, conic or even-asphere, an even-asphere
            with a non-zero r^2 term or a non-zero term above r^20, a
            non-refractive interaction model (thin-lens,
            diffractive, phase), an aspheric or conic image surface,
            ideal-material absorption, a ``DataMaterial``, or a field
            definition with no CODE V equivalent.
        ValueError: If any number to be written is not finite (``inf`` or
            ``nan``; an infinite object distance and an infinite ``r_max`` are
            allowed), the object thickness is ``-inf``, a surface aperture has
            an invalid radius (``r_max <= r_min``, negative or ``nan``), the
            wavelength weights do not match the wavelengths, the optic has more
            than 21 wavelengths or more than 25 fields (the most CODE V
            loads), a material needing a private glass has no positive index at
            one of the system wavelengths, a wavelength is zero or negative, no
            wavelength is primary, an optic with a real surface has no stop
            surface, no aperture, no fields or no wavelengths, the aperture is
            ``objectNA`` of 1 or more, or the object distance is exactly 0.

    Warns:
        UserWarning: For a material with no verified CODE V catalog name, or a
            catalog glass with a system wavelength outside its data range (both
            written as a private glass with its index at the system
            wavelengths); for extinction ``k`` of such a glass (not exported);
            for pickups or solves (their resolved values are exported); for a
            surface aperture other than a plain ``RadialAperture``, or any
            aperture on the object or image surface (not exported); for field
            vignetting factors ``vx``/``vy`` (not exported); for a
            ``float_by_stop_size`` aperture, or an ``imageFNO`` aperture with
            the object at a finite distance (both written as a fixed ``EPD``);
            and for an unknown aperture type (aperture skipped).
    """
    model = OpticToCodeVConverter(optic).convert()
    lines = CodeVFileEncoder(model).encode()
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


class CodeVWriter(BaseOpticWriter):
    """BaseOpticWriter implementation for CODE V .seq files.

    This thin wrapper around :func:`save_codev_file` allows the CODE V writer
    to be used polymorphically via the BaseOpticWriter interface.
    """

    def write(self, optic: Optic, filepath: str) -> list[str]:
        """Write *optic* to a .seq file at *filepath*.

        Args:
            optic: The optic to export.
            filepath: Destination path.

        Returns:
            An empty list (warnings are issued via the ``warnings`` module).
        """
        save_codev_file(optic, filepath)
        return []
