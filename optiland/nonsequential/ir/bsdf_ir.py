"""BsdfIR -- data-only description of a surface scatter model.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Every BaseBSDF subclass this revamp lowers, plus "none" for a bare
# specular/refractive/absorbing surface with no attached scatter lobe.
# PR9 (BSDF lobes) restructures BaseBSDF around explicit REFLECT/TRANSMIT
# lobes; this kind list is the pre-PR9 physics, carried through as data.
BsdfKind = Literal["none", "specular", "lambertian", "harvey_shack", "tabulated"]


@dataclass(frozen=True)
class BsdfIR:
    """A surface scatter model, as plain data.

    ``params`` is kind-specific and mirrors the corresponding ``BaseBSDF``
    subclass's constructor arguments exactly, so lowering is a direct field
    copy with no interpretation:

    - ``"none"``: ``{}``
    - ``"specular"``: ``{}``
    - ``"lambertian"``: ``{"reflectance_value": <float>}``
    - ``"harvey_shack"``: ``{"b0": <float>, "l0": <float>, "s": <float>}``
    - ``"tabulated"``: ``{"path": <str>}``

    Attributes:
        kind: Which scatter model this is.
        params: Kind-specific parameters (see above).
    """

    kind: BsdfKind
    params: dict[str, Any] = field(default_factory=dict)
