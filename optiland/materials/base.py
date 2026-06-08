"""Base Material

This module defines the base class for materials. The base class provides
methods to calculate the refractive index, extinction coefficient, and Abbe
number of a material. Subclasses of BaseMaterial should implement the `n` and
`k` methods to provide specific material properties.

Kramer Harrison, 2024
"""

from __future__ import annotations

import hashlib
import weakref
from abc import ABC, abstractmethod

import numpy as np

import optiland.backend as be
from optiland.propagation.base import BasePropagationModel
from optiland.propagation.homogeneous import HomogeneousPropagation

# Maps id(array) -> (weakref, content_digest, version_token) so the O(N) content
# hash in BaseMaterial._array_metadata_key runs once per array object and is
# reused on later lookups. The weakref callback evicts the entry when the array
# is collected, so a later array reusing the same id() never reads a stale
# digest.
_ARRAY_DIGEST_CACHE: dict[int, tuple] = {}


def _array_content_key(value, digest: bytes) -> tuple:
    """Assemble a cache-key tuple from a content digest plus coarse metadata."""
    return (
        "array-content",
        digest,
        tuple(getattr(value, "shape", ())),
        str(getattr(value, "dtype", type(value).__name__)),
        str(getattr(value, "device", None)),
    )


class BaseMaterial(ABC):
    """Base class for materials.

    This class defines the interface for material properties such as
    refractive index (n) and extinction coefficient (k). It also provides a
    method to calculate the Abbe number.

    Subclasses of BaseMaterial should implement the abstract methods `n` and
    `k` to provide specific material properties.

    Attributes:
        propagation_model: The model used to propagate rays through this
            material.

    Methods:
        n(wavelength: float | be.ndarray) -> float | be.ndarray:
            Abstract method to calculate the refractive index at a given
            wavelength(s) in microns.
        k(wavelength: float | be.ndarray) -> float | be.ndarray:
            Abstract method to calculate the extinction coefficient at a given
            wavelength(s) in microns.
        abbe() -> float:
            Method to calculate the Abbe number of the material.

    """

    _registry = {}
    _MAX_VALUE_KEY_ARRAY_SIZE = 1024

    def __init__(self, propagation_model: BasePropagationModel | None = None):
        """Initializes the material and its caches.

        Args:
            propagation_model: The propagation model to use for this material.
                If None, a default HomogeneousPropagation model is created.
        """
        self._n_cache = {}
        self._k_cache = {}

        if propagation_model is None:
            self.propagation_model = HomogeneousPropagation(self)
        else:
            self.propagation_model = propagation_model

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseMaterial._registry[cls.__name__] = cls

    def __eq__(self, value: object) -> bool:
        return isinstance(value, type(self)) and value.to_dict() == self.to_dict()

    @classmethod
    def _array_size(cls, value) -> int | None:
        """Return the total element count for array-like values if available."""
        shape = getattr(value, "shape", None)
        if shape is None:
            return None

        try:
            return int(np.prod(shape, dtype=np.int64))
        except Exception:
            return None

    @staticmethod
    def _array_metadata_key(value) -> tuple:
        """Build a content-addressed cache key for large arrays.

        The key identifies an array by its *contents*, never by its memory
        location. Raw buffer pointers (``ndarray.__array_interface__["data"]``,
        ``Tensor.data_ptr()``) and ``id()`` are reused by the allocator once an
        array is garbage-collected, so two distinct wavelength arrays of the
        same shape/dtype that reuse a freed slot would collide and ``n()`` would
        return the previous array's refractive index -- a silent
        cross-wavelength leak (issue #630).

        Hashing the bytes is O(N), so the digest is memoized per array object in
        ``_ARRAY_DIGEST_CACHE``; repeated lookups of the same array (e.g. one
        wavelength bundle traced through every surface) are amortized O(1). A
        weakref callback drops the entry when the array dies, so id() reuse can
        never surface a stale digest.

        NumPy exposes no in-place-write counter, so an array is treated as
        immutable for its lifetime (optiland never mutates wavelength buffers in
        place). Torch tensors carry ``_version``, folded into the memoization
        token so in-place edits invalidate the cached digest.
        """
        oid = id(value)
        token = int(getattr(value, "_version", 0))
        cached = _ARRAY_DIGEST_CACHE.get(oid)
        if cached is not None:
            ref, digest, cached_token = cached
            if ref() is value and cached_token == token:
                return _array_content_key(value, digest)

        if hasattr(value, "detach"):  # torch tensor
            array = value.detach().cpu().contiguous().numpy()
        else:  # numpy ndarray, list, or tuple
            array = np.ascontiguousarray(np.asarray(value))
        digest = hashlib.blake2b(array.tobytes(), digest_size=16).digest()

        try:
            ref = weakref.ref(
                value, lambda _ref, _oid=oid: _ARRAY_DIGEST_CACHE.pop(_oid, None)
            )
        except TypeError:
            ref = None  # e.g. list/tuple cannot be weak-referenced; skip memo
        if ref is not None:
            _ARRAY_DIGEST_CACHE[oid] = (ref, digest, token)
        return _array_content_key(value, digest)

    def _create_cache_key(self, wavelength: float | be.ndarray, **kwargs) -> tuple:
        """Creates a hashable cache key from wavelength and kwargs."""
        if be.is_array_like(wavelength):
            size = self._array_size(wavelength)
            if size is not None and size <= self._MAX_VALUE_KEY_ARRAY_SIZE:
                wavelength_key = tuple(np.ravel(be.to_numpy(wavelength)))
            else:
                wavelength_key = self._array_metadata_key(wavelength)
        else:
            wavelength_key = wavelength
        return (wavelength_key,) + tuple(sorted(kwargs.items()))

    @staticmethod
    def _requires_grad(value) -> bool:
        """Check if a value is a torch tensor that requires gradient."""
        return hasattr(value, "requires_grad") and value.requires_grad

    @staticmethod
    def _detach_if_tensor(value):
        """Detach a torch tensor to sever the computation graph link.

        This prevents the 'backward through the graph a second time' error
        that occurs when a cached tensor still references a freed computation
        graph.
        """
        if hasattr(value, "detach"):
            return value.detach()
        return value

    def n(self, wavelength: float | be.ndarray, **kwargs) -> float | be.ndarray:
        """Calculates the refractive index at a given wavelength with caching.

        Args:
            wavelength (float | be.ndarray): The wavelength(s) of light in microns.
                Can be a float, numpy array, or torch tensor.
            **kwargs: Additional keyword arguments for calculation (e.g., temperature).

        Returns:
            float | be.ndarray: The refractive index at the given wavelength(s).
        """
        cache_key = self._create_cache_key(wavelength, **kwargs)

        if cache_key in self._n_cache:
            return self._n_cache[cache_key]

        result = self._calculate_n(wavelength, **kwargs)

        # If the result requires grad, it is connected to an optimization
        # variable (e.g. the index itself is being optimized).  In that case
        # we must NOT cache — every forward pass needs a fresh graph.
        if self._requires_grad(result):
            return result

        # Otherwise the value is a constant w.r.t. optimization variables.
        # Detach before caching to avoid holding a stale computation graph.
        self._n_cache[cache_key] = self._detach_if_tensor(result)
        return self._n_cache[cache_key]

    def k(self, wavelength: float | be.ndarray, **kwargs) -> float | be.ndarray:
        """Calculates the extinction coefficient at a given wavelength with caching.

        Args:
            wavelength (float | be.ndarray): The wavelength(s) of light in microns.
                Can be a float, numpy array, or torch tensor.
            **kwargs: Additional keyword arguments for calculation.

        Returns:
            float | be.ndarray: The extinction coefficient at the given wavelength(s).
        """
        cache_key = self._create_cache_key(wavelength, **kwargs)

        if cache_key in self._k_cache:
            return self._k_cache[cache_key]

        result = self._calculate_k(wavelength, **kwargs)
        # Same logic as n(): skip cache if result is differentiable.
        if self._requires_grad(result):
            return result

        self._k_cache[cache_key] = self._detach_if_tensor(result)
        return self._k_cache[cache_key]

    @abstractmethod
    def _calculate_n(
        self, wavelength: float | be.ndarray, **kwargs
    ) -> float | be.ndarray:
        """Calculates the refractive index at a given wavelength.

        Args:
            wavelength (float | be.ndarray): The wavelength(s) of light in microns.

        Returns:
            float | be.ndarray: The refractive index at the given wavelength(s).
        """
        pass  # pragma: no cover

    @abstractmethod
    def _calculate_k(
        self, wavelength: float | be.ndarray, **kwargs
    ) -> float | be.ndarray:
        """Calculates the extinction coefficient at a given wavelength.

        Args:
            wavelength (float | be.ndarray): The wavelength(s) of light in microns.

        Returns:
            float | be.ndarray: The extinction coefficient at the given
            wavelength(s).
        """
        pass  # pragma: no cover

    def abbe(self) -> float:
        """Calculate the Abbe number (Vd) of the material.

        The Abbe number is a measure of the material's dispersion, defined as
        Vd = (n_d - 1) / (n_F - n_C), where n_d, n_F, and n_C are the
        refractive indices at the Fraunhofer d (587.5618 nm), F (486.1327 nm),
        and C (656.2725 nm) spectral lines, respectively.

        Returns:
            float: The Abbe number of the material.

        """
        nD = self.n(0.5875618)
        nF = self.n(0.4861327)
        nC = self.n(0.6562725)
        return (nD - 1) / (nF - nC)

    def to_dict(self):
        """Convert the material to a dictionary.

        Returns:
            dict: The dictionary representation of the material.

        """
        return {
            "type": self.__class__.__name__,
            "propagation_model": self.propagation_model.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        """Create a material from a dictionary representation.

        This factory method first delegates to the appropriate subclass to
        create the material instance, then handles the deserialization of
        the propagation model.

        Args:
            data (dict): The dictionary representation of the material.

        Returns:
            BaseMaterial: An instance of a specific material subclass created
            from the dictionary data.

        """
        material_type = data.get("type")
        if material_type not in cls._registry:
            raise ValueError(f"Unknown material type: {material_type}")

        # Delegate to the correct subclass to create the instance.
        material_subclass = cls._registry[material_type]
        material = material_subclass.from_dict(data)

        # Handle the propagation model deserialization here.
        propagation_model_data = data.get("propagation_model")
        if propagation_model_data:
            # Create the model, passing the material to resolve dependencies.
            new_prop_model = BasePropagationModel.from_dict(
                propagation_model_data, material=material
            )
            # Overwrite the default propagation model.
            material.propagation_model = new_prop_model

        return material
