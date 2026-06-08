"""Optiland File Handler

This module provides functions for loading and saving Optiland objects and
Optic instances to and from JSON files. The functions `load_obj_from_json` and
`save_obj_to_json` can be used to load and save any object that has a class
method `from_dict` and a method `to_dict`, respectively. In Optiland, this
includes most core classes, such as Optic, BaseGeometry, BaseCoating,
BaseMaterial, Aperture, FieldGroup, WavelengthGroup, etc.

Kramer Harrison, 2024
"""

from __future__ import annotations

import json
import os

from optiland.optic import Optic


def load_obj_from_json(cls, filepath):
    """Load an object from a JSON file.

    Note that this function can be used to load any object that has a class
    method `from_dict` that can be used to create an instance of the class from
    a dictionary. In Optiland, this includes most core classes, such as Optic,
    BaseGeometry, BaseCoating, BaseMaterial, Aperture, FieldGroup,
    WavelengthGroup, etc.

    Args:
        cls: The class of the object to load.
        filepath: The path to the JSON file.

    Returns:
        An instance of the class

    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' does not exist.")
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return cls.from_dict(data)


class OptilandEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle tensor/array serialization across backends."""

    def default(self, obj):
        if hasattr(obj, "tolist") and callable(obj.tolist):
            return obj.tolist()
        elif hasattr(obj, "item") and callable(obj.item):
            return obj.item()
        return super().default(obj)


def save_obj_to_json(obj, filepath):
    """Save an object to a JSON file.

    Note that this function can be used to save any object that has a method
    `to_dict` that returns a dictionary representation of the object. In
    Optiland, this includes most core classes, such as Optic, BaseGeometry,
    BaseCoating, BaseMaterial, Aperture, FieldGroup, WavelengthGroup, etc.

    Args:
        obj: The object to save.
        filepath: The path to the JSON file.

    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj.to_dict(), f, indent=4, cls=OptilandEncoder)


def load_optiland_file(filepath):
    """Load an Optiland Optic from a JSON file.

    Args:
        filepath: The path to the JSON file.

    Returns:
        An Optic instance.

    """
    return load_obj_from_json(Optic, filepath)


def save_optiland_file(obj, filepath):
    """Save an Optiland Optic to a JSON file.

    Args:
        obj: The Optic to save.
        filepath: The path to the JSON file.

    """
    save_obj_to_json(obj, filepath)
