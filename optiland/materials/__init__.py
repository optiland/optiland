"""This package defines material types used in Optiland, including ideal
materials, materials based on Abbe numbers, and materials loaded from
refractiveindex.info data files."""

from __future__ import annotations

from .abbe import AbbeMaterial, AbbeMaterialE
from .base import BaseMaterial
from .ideal import IdealMaterial
from .material import Material
from .material_file import MaterialFile
from .material_utils import (
    downsample_glass_map,
    find_closest_glass,
    get_nd_vd,
    get_neighbour_glasses,
    glasses_selection,
    plot_glass_map,
    plot_nk,
)
from .model import ModelMaterial

__all__ = [
    # From abbe.py
    "AbbeMaterial",
    "AbbeMaterialE",
    # From base.py
    "BaseMaterial",
    # From ideal.py
    "IdealMaterial",
    # From material.py
    "Material",
    # From material_file.py
    "MaterialFile",
    # From material_utils.py
    "downsample_glass_map",
    "get_nd_vd",
    "get_neighbour_glasses",
    "glasses_selection",
    "plot_glass_map",
    "plot_nk",
    "find_closest_glass",
    "ModelMaterial",
]
