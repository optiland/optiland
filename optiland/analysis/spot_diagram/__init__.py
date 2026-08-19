"""Spot Diagram Analysis Package

This package provides spot diagram analysis for optical systems, including
configurable reference centering strategies.
"""

from __future__ import annotations

from .core import SpotData, SpotDiagram
from .reference import SpotReferenceType

__all__ = ["SpotData", "SpotDiagram", "SpotReferenceType"]
