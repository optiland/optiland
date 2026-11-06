"""Resolve static OSLO prescription pickups in surface order."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any

from optiland.fileio.oslo.reader.coordinates import reference_index


def resolve_pickups(surfaces: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Resolve documented preceding-surface pickups without mutating the input.

    PK CV/CVM copy the full profile with an additive curvature offset. TH/THM
    and LN/LNM have additive length offsets (never multiplicative factors).
    Forward and cyclic references are invalid OSLO static pickups.
    """
    resolved = deepcopy(surfaces)
    for index, data in sorted(resolved.items()):
        for pickup in data.get("pickups", []):
            kind = pickup[0].upper()
            source = reference_index(int(pickup[1]), index)
            if source not in resolved or not 0 <= source < index:
                raise ValueError(
                    f"OSLO PK at surface {index} requires a preceding source"
                )
            original = resolved[source]
            negative = kind in {"CVM", "THM", "LNM", "TDM"}
            sign = -1 if negative else 1
            offset = (
                float(pickup[2])
                if len(pickup) > 2 and kind not in {"LN", "LNM"}
                else 0.0
            )
            if kind in {"CV", "CVM"}:
                for key in list(data):
                    if key in {
                        "CC",
                        "CVX",
                        "ASP",
                        "AD",
                        "AE",
                        "AF",
                        "AG",
                    } or re.fullmatch(r"AS\d+", key):
                        del data[key]
                for key, value in original.items():
                    if key in {"CC", "ASP"}:
                        data[key] = value
                    elif key in {"CVX", "AD", "AE", "AF", "AG"} or re.fullmatch(
                        r"AS\d+", key
                    ):
                        data[key] = sign * value
                radius = original.get("RD", math.inf)
                curvature = (sign / radius if radius else 0.0) + offset
                data["RD"] = 1 / curvature if curvature else math.inf
            elif kind in {"TH", "THM"}:
                data["TH"] = sign * original.get("TH", 0.0) + offset
            elif kind in {"LN", "LNM"}:
                end = (
                    reference_index(int(pickup[2]), index) if int(pickup[2]) else index
                )
                if not source < end <= index:
                    raise ValueError(f"OSLO PK {kind} has an invalid length range")
                offset = float(pickup[3]) if len(pickup) > 3 else 0.0
                data["TH"] = (
                    sign * sum(resolved[i].get("TH", 0.0) for i in range(source, end))
                    + offset
                )
            elif kind == "AP":
                data["AP"] = original.get("AP", 0.0)
            elif kind == "GLA":
                data["material"] = original.get("material", "AIR")
                if data["material"] == "RFL":
                    raise ValueError("OSLO PK GLA cannot pick up a reflector")
                if "glass_wavelengths" in original:
                    data["glass_wavelengths"] = list(original["glass_wavelengths"])
            elif kind in {"TD", "TDM"}:
                if any(key in original or key in data for key in ("GC", "RCO", "BEN")):
                    raise ValueError(
                        "OSLO PK TD/TDM with global/return/bend data is not mapped"
                    )
                for key in ("DCX", "DCY", "DCZ", "TLA", "TLB", "TLC"):
                    data[key] = sign * original.get(key, 0.0)
                for key in ("TOX", "TOY", "TOZ"):
                    data[key] = original.get(key, 0.0)
                data["DT"] = sign * original.get("DT", 1)
            else:
                raise ValueError(f"OSLO PK {kind} is unsupported")
        for target, source_value, source_id in data.get("aperture_pickups", []):
            source = reference_index(int(source_value), index)
            if source not in resolved or source >= index or source < 0:
                raise ValueError(
                    f"OSLO APK at surface {index} requires a preceding source"
                )
            apertures = resolved[source].get("special_apertures", {})
            if source_id not in apertures:
                raise ValueError(f"OSLO APK source aperture {source_id} is undefined")
            data.setdefault("special_apertures", {})[target] = deepcopy(
                apertures[source_id]
            )
    return resolved
