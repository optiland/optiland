"""AGF Material

This module reads glass catalogs in the AGF format used by OpticStudio and
evaluates their glasses with the catalog's own dispersion formula and
coefficients. No catalog data ship with Optiland: the reader works on AGF
files the user already has.

An AGF file is a list of records, one per line, keyed by their first token.
The reader uses these:

- ``NM name formula MIL nd vd ...``: starts a glass and names its formula.
- ``CD c1 c2 ...``: the dispersion coefficients, for wavelengths in µm.
- ``LD min max``: the wavelength range of the data, in µm.

The other records (thermal, transmission, comments) are skipped. Indices are
relative to air, as in the catalogs, and are evaluated at the catalog's
reference temperature; the thermal coefficients on the ``TD`` line are not
applied.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.materials.base import BaseMaterial

if TYPE_CHECKING:
    from optiland.propagation.base import BasePropagationModel


# OpticStudio formula codes, in the order its glass catalog lists them.
FORMULA_NAMES = {
    1: "Schott",
    2: "Sellmeier 1",
    3: "Herzberger",
    4: "Sellmeier 2",
    5: "Conrady",
    6: "Sellmeier 3",
    7: "Handbook of Optics 1",
    8: "Handbook of Optics 2",
    9: "Sellmeier 4",
    10: "Extended",
    11: "Sellmeier 5",
    12: "Extended 2",
    13: "Extended 3",
}

# Coefficients each formula reads from the CD line.
_COEFFICIENT_COUNTS = {
    1: 6,
    2: 6,
    3: 6,
    4: 5,
    5: 3,
    6: 8,
    7: 4,
    8: 4,
    9: 5,
    10: 8,
    11: 10,
    12: 8,
    13: 9,
}


def _sellmeier(c, w2, terms: int):
    """Sum of K w²/(w² - L) terms, with c = (K1, L1, K2, L2, ...)."""
    total = 0.0
    for i in range(terms):
        total = total + c[2 * i] * w2 / (w2 - c[2 * i + 1])
    return total


def evaluate_agf_formula(formula: int, c, wavelength):
    """Refractive index from an OpticStudio dispersion formula.

    Args:
        formula: The formula code, 1 to 13 (see ``FORMULA_NAMES``).
        c: The coefficients, in the order of the AGF ``CD`` line.
        wavelength: Wavelength(s) in µm.

    Returns:
        The refractive index relative to air.
    """
    if formula not in _COEFFICIENT_COUNTS:
        raise ValueError(f"Unknown AGF dispersion formula {formula!r}")
    if len(c) < _COEFFICIENT_COUNTS[formula]:
        raise ValueError(
            f"AGF formula {formula} ({FORMULA_NAMES[formula]}) needs "
            f"{_COEFFICIENT_COUNTS[formula]} coefficients, got {len(c)}"
        )
    w = wavelength
    w2 = w * w

    if formula == 1:  # Schott
        n2 = c[0] + c[1] * w2 + c[2] / w2 + c[3] / w2**2 + c[4] / w2**3 + c[5] / w2**4
    elif formula == 2:  # Sellmeier 1
        n2 = 1.0 + _sellmeier(c, w2, 3)
    elif formula == 3:  # Herzberger
        L = 1.0 / (w2 - 0.028)
        return c[0] + c[1] * L + c[2] * L**2 + c[3] * w2 + c[4] * w2**2 + c[5] * w2**3
    elif formula == 4:  # Sellmeier 2
        n2 = 1.0 + c[0] + c[1] * w2 / (w2 - c[2] ** 2) + c[3] / (w2 - c[4] ** 2)
    elif formula == 5:  # Conrady
        return c[0] + c[1] / w + c[2] / w**3.5
    elif formula == 6:  # Sellmeier 3
        n2 = 1.0 + _sellmeier(c, w2, 4)
    elif formula == 7:  # Handbook of Optics 1
        n2 = c[0] + c[1] / (w2 - c[2]) - c[3] * w2
    elif formula == 8:  # Handbook of Optics 2
        n2 = c[0] + c[1] * w2 / (w2 - c[2]) - c[3] * w2
    elif formula == 9:  # Sellmeier 4
        n2 = c[0] + c[1] * w2 / (w2 - c[2]) + c[3] * w2 / (w2 - c[4])
    elif formula == 10:  # Extended
        n2 = (
            c[0]
            + c[1] * w2
            + c[2] / w2
            + c[3] / w2**2
            + c[4] / w2**3
            + c[5] / w2**4
            + c[6] / w2**5
            + c[7] / w2**6
        )
    elif formula == 11:  # Sellmeier 5
        n2 = 1.0 + _sellmeier(c, w2, 5)
    elif formula == 12:  # Extended 2
        n2 = (
            c[0]
            + c[1] * w2
            + c[2] / w2
            + c[3] / w2**2
            + c[4] / w2**3
            + c[5] / w2**4
            + c[6] * w2**2
            + c[7] * w2**3
        )
    else:  # 13, Extended 3
        n2 = (
            c[0]
            + c[1] * w2
            + c[2] * w2**2
            + c[3] / w2
            + c[4] / w2**2
            + c[5] / w2**3
            + c[6] / w2**4
            + c[7] / w2**5
            + c[8] / w2**6
        )
    return be.sqrt(n2)


@dataclass
class AGFGlass:
    """One glass record of an AGF catalog."""

    name: str
    formula: int
    coefficients: list[float]
    nd: float | None = None
    vd: float | None = None
    wavelength_range: tuple[float, float] | None = None
    catalog: str | None = None
    comment: str = field(default="", repr=False)


def _read_text(path: str) -> str:
    """Read an AGF file, which OpticStudio writes as ASCII or UTF-16."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _floats(tokens: list[str]) -> list[float]:
    values = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError:
            values.append(0.0)
    return values


def read_agf(path: str) -> dict[str, AGFGlass]:
    """Read the glasses of an AGF catalog.

    Args:
        path: The AGF file.

    Returns:
        The glasses keyed by name, in file order. The catalog name is the
        file name without its extension, as OpticStudio uses it in GCAT.
    """
    catalog = os.path.splitext(os.path.basename(path))[0]
    glasses: dict[str, AGFGlass] = {}
    glass = None
    for line in _read_text(path).splitlines():
        tokens = line.split()
        if not tokens:
            continue
        key = tokens[0].upper()
        if key == "NM" and len(tokens) >= 3:
            values = _floats(tokens[2:6])
            glass = AGFGlass(
                name=tokens[1],
                formula=int(values[0]),
                coefficients=[],
                nd=values[2] if len(values) > 2 else None,
                vd=values[3] if len(values) > 3 else None,
                catalog=catalog,
            )
            glasses[glass.name] = glass
        elif glass is None:
            continue
        elif key == "CD":
            glass.coefficients = _floats(tokens[1:])
        elif key == "LD" and len(tokens) >= 3:
            low, high = _floats(tokens[1:3])
            glass.wavelength_range = (low, high)
        elif key == "GC":
            glass.comment = line.strip()[2:].strip()
    return glasses


AGF_PATH_VARIABLE = "OPTILAND_AGF_PATH"


def agf_search_dirs(agf_dirs=None) -> list[str]:
    """The folders to search for AGF catalogs.

    Args:
        agf_dirs: A folder or a list of folders. If None, the folders in the
            ``OPTILAND_AGF_PATH`` environment variable, separated by
            ``os.pathsep``, are used; if that is unset too, there are none.

    Returns:
        The folders, in search order.
    """
    if agf_dirs is None:
        value = os.environ.get(AGF_PATH_VARIABLE, "")
        return [d for d in value.split(os.pathsep) if d]
    if isinstance(agf_dirs, str | os.PathLike):
        return [os.fspath(agf_dirs)]
    return [os.fspath(d) for d in agf_dirs]


def find_agf_catalog(catalog: str, agf_dirs: list[str]) -> str | None:
    """The AGF file of a catalog, matching the file name ignoring case.

    The first folder that has the catalog wins, as for a search path.
    """
    target = catalog.lower() + ".agf"
    for directory in agf_dirs:
        if not os.path.isdir(directory):
            continue
        for entry in sorted(os.listdir(directory)):
            if entry.lower() == target:
                return os.path.join(directory, entry)
    return None


class AGFMaterial(BaseMaterial):
    """A glass evaluated from its AGF catalog entry.

    The index comes from the catalog's own dispersion formula and
    coefficients, as OpticStudio evaluates it, relative to air at the
    catalog's reference temperature. The extinction coefficient is zero.

    Args:
        name: The glass name.
        formula: The OpticStudio formula code, 1 to 13.
        coefficients: The coefficients of the AGF ``CD`` line.
        catalog: The catalog the glass came from, if known.
        wavelength_range: The (min, max) wavelength of the data in µm.
        propagation_model: The propagation model; homogeneous if None.
    """

    def __init__(
        self,
        name: str,
        formula: int,
        coefficients,
        catalog: str | None = None,
        wavelength_range: tuple[float, float] | None = None,
        propagation_model: BasePropagationModel | None = None,
    ):
        super().__init__(propagation_model)
        self.name = name
        self.formula = int(formula)
        self.coefficients = [float(c) for c in coefficients]
        self.catalog = catalog
        self.wavelength_range = (
            tuple(float(w) for w in wavelength_range) if wavelength_range else None
        )
        # Fail now on a bad record, not at the first trace.
        evaluate_agf_formula(self.formula, self.coefficients, 0.5875618)

    @classmethod
    def from_glass(cls, glass: AGFGlass) -> AGFMaterial:
        """Create the material from a record returned by ``read_agf``."""
        return cls(
            glass.name,
            glass.formula,
            glass.coefficients,
            catalog=glass.catalog,
            wavelength_range=glass.wavelength_range,
        )

    @classmethod
    def from_file(cls, path: str, name: str) -> AGFMaterial:
        """Create the material for glass ``name`` in the AGF file ``path``.

        The name must match exactly, ignoring case.
        """
        glasses = read_agf(path)
        glass = glasses.get(name)
        if glass is None:
            lowered = {key.lower(): value for key, value in glasses.items()}
            glass = lowered.get(name.lower())
        if glass is None:
            raise ValueError(f"Glass {name!r} is not in {path}")
        return cls.from_glass(glass)

    @property
    def display_name(self) -> str:
        return self.name

    def spectral_range(self, property_name: str = "n") -> tuple[float, float] | None:
        super().spectral_range(property_name)
        return self.wavelength_range

    def _cache_state(self) -> tuple | None:
        """Track the formula and coefficients, including in-place edits."""
        return self._state_key((self.formula, self.coefficients))

    def _calculate_n(self, wavelength, **kwargs):
        wavelength = self._as_backend_array(wavelength)
        return evaluate_agf_formula(self.formula, self.coefficients, wavelength)

    def _calculate_k(self, wavelength, **kwargs):
        if be.is_array_like(wavelength) and be.size(wavelength) > 1:
            return be.zeros_like(self._as_backend_array(wavelength))
        return be.array(0.0)

    def to_dict(self):
        material_dict = super().to_dict()
        material_dict.update(
            {
                "name": self.name,
                "formula": self.formula,
                "coefficients": list(self.coefficients),
                "catalog": self.catalog,
                "wavelength_range": (
                    list(self.wavelength_range) if self.wavelength_range else None
                ),
            }
        )
        return material_dict

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["name"],
            data["formula"],
            data["coefficients"],
            catalog=data.get("catalog"),
            wavelength_range=data.get("wavelength_range"),
        )
