"""BsdfIR -- data-only description of a surface scatter model.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Every BaseBSDF subclass this revamp lowers, plus "none" for a bare
# specular/refractive/absorbing surface with no attached scatter lobe.
# BaseBSDF is restructured around explicit REFLECT/TRANSMIT lobes:
# sample() returns which side of the surface each scattered ray landed on,
# so RefractiveComponent can resolve n_current/k_current from the lobe's own
# choice instead of the independent Fresnel branch draw. The kind list
# itself is unchanged by that -- only params gained transmissive_fraction.
BsdfKind = Literal["none", "specular", "lambertian", "harvey_shack", "tabulated"]


@dataclass(frozen=True)
class BsdfIR:
    """A surface scatter model, as plain data.

    ``params`` is kind-specific and mirrors the corresponding ``BaseBSDF``
    subclass's constructor arguments exactly, so lowering is a direct field
    copy with no interpretation:

    - ``"none"``: ``{}``
    - ``"specular"``: ``{}`` (always reflective; no transmissive lobe)
    - ``"lambertian"``: ``{"reflectance_value": <float>,
      "transmissive_fraction": <float>}``
    - ``"harvey_shack"``: ``{"b0": <float>, "l0": <float>, "s": <float>,
      "transmissive_fraction": <float>}``
    - ``"tabulated"``: ``{"path": <str>, "transmissive_fraction": <float>}``

    ``transmissive_fraction`` is the probability that a given scatter
    event samples the transmissive (far-side) hemisphere instead of the
    reflective one; it defaults to 0.0 on every kind that has it, so an
    un-set BSDF scatters exactly as it did before D-5.

    Attributes:
        kind: Which scatter model this is.
        params: Kind-specific parameters (see above).
    """

    kind: BsdfKind
    params: dict[str, Any] = field(default_factory=dict)
