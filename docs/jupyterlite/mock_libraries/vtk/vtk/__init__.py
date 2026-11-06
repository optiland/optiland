"""
VTK WASM Compatibility Stub
This package intercepts and stubs all VTK submodules to prevent
ImportErrors in environments where native VTK is unavailable.
"""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from unittest.mock import MagicMock


class DeepStubModule(ModuleType):
    """A virtual module that generates stub sub-modules or attributes dynamically."""

    def __getattr__(self, name):
        # Prevent infinite recursion when Python looks up internal/dunder strings
        if name.startswith("__"):
            raise AttributeError(name)

        # Construct the expected full package path string
        fullname = f"{self.__name__}.{name}"

        # If the submodule has already been queried, return it instantly
        if fullname in sys.modules:
            return sys.modules[fullname]

        # If the attribute has not been queried already, register and return a new
        # stub module
        sub_mod = DeepStubModule(fullname)
        sys.modules[fullname] = sub_mod
        return sub_mod

    def __call__(self, *args, **kwargs):
        # Allows initializing arbitrary VTK components safely, e.g. vtk.vtkActor()
        return MagicMock()

    def load_module(self, fullname):
        # Fallback for older import machinery if needed
        return sys.modules[fullname]

    def exec_module(self, module):
        # Required by importlib.machinery.ModuleSpec loader interface
        pass


self_module = sys.modules[__name__]
self_module.__class__ = DeepStubModule
self_module.__path__ = []


class VTKSubmoduleFinder:
    def find_spec(self, fullname, path, target=None):
        # Only intercept lookups explicitly prefixed with 'vtk.'
        if fullname.startswith("vtk."):
            if fullname in sys.modules:
                sub_mod = sys.modules[fullname]
            else:
                # Otherwise, dynamically spawn it now
                parts = fullname.split(".")
                parent = sys.modules[".".join(parts[:-1])]
                sub_mod = getattr(parent, parts[-1])

            return ModuleSpec(fullname, sub_mod, is_package=True)
        return None


sys.meta_path.insert(0, VTKSubmoduleFinder())
