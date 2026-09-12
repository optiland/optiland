"""Base Material

This module defines the base class for materials. The base class provides
methods to calculate the refractive index, extinction coefficient, and Abbe
number of a material. Subclasses implement `_calculate_n` and `_calculate_k`;
the public methods manage evaluation and optional caching.

Kramer Harrison, 2024
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

import numpy as np

import optiland.backend as be
from optiland._suggest import options_hint
from optiland.propagation.base import BasePropagationModel
from optiland.propagation.homogeneous import HomogeneousPropagation

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None


def _array_content_key(value, digest: bytes) -> tuple:
    """Assemble a cache-key tuple from a content digest plus coarse metadata."""
    return (
        "array-content",
        digest,
        tuple(getattr(value, "shape", ())),
        str(getattr(value, "dtype", type(value).__name__)),
        str(getattr(value, "device", None)),
    )


def _array_uniform_key(value, scalar: float) -> tuple:
    """Assemble a cache-key tuple for an array whose elements are all equal."""
    return (
        "array-uniform",
        scalar,
        tuple(getattr(value, "shape", ())),
        str(getattr(value, "dtype", type(value).__name__)),
        str(getattr(value, "device", None)),
    )


def _uniform_scalar(value) -> float | None:
    """Return the common value of a constant array, or None if not constant.

    Ray bundles traced at a single wavelength carry that wavelength repeated
    per ray, so the by-far most common "large array" seen here is a broadcast
    constant. Detecting it costs one on-device reduction instead of the full
    device-to-host copy plus O(N) hash of the content-digest path, and lets
    the property caches store a single value instead of an N-element result.
    """
    try:
        if torch is not None and isinstance(value, torch.Tensor):
            # The key only needs the numeric value; detaching avoids the
            # scalar-conversion warning for grad-attached bundles and keeps
            # the uniformity probe off the autograd graph.
            detached = value.detach()
            first = detached.reshape(-1)[0]
            if bool(torch.eq(detached, first).all()):
                return float(first)
            return None
        array = np.asarray(value)
        first = array.flat[0]
        if np.all(array == first):
            return float(first)
        return None
    except (IndexError, TypeError, ValueError, RuntimeError):
        return None


class BaseMaterial(ABC):
    """Base class for materials.

    This class defines the interface for material properties such as
    refractive index (n) and extinction coefficient (k). It also provides a
    method to calculate the Abbe number.

    Subclasses implement `_calculate_n` and `_calculate_k`. Result caching is
    optional: override `_cache_state` only when all optical state can be tracked.

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
        self._cache_context = None

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
            return math.prod(shape)
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

        Inspect caller-owned storage on every lookup. NumPy has no mutation
        counter, Torch inference tensors are mutable without a version counter,
        and writes through a NumPy alias do not increment a Torch version.
        Identity/version memoization therefore cannot guarantee correctness.

        Constant arrays -- the overwhelmingly common case, since a ray bundle
        traced at one wavelength repeats that wavelength per ray -- are detected
        first and keyed by their single value, skipping the device-to-host copy
        and O(N) hash entirely while remaining content-addressed.
        """
        # Lists and tuples have no shape/dtype attributes. Normalize only the
        # key input so nested and flat sequences cannot share a fingerprint.
        if isinstance(value, (list, tuple)):
            value = np.asarray(value)
        scalar = _uniform_scalar(value)
        if scalar is not None:
            key = _array_uniform_key(value, scalar)
        else:
            if hasattr(value, "detach"):  # torch tensor
                array = value.detach().cpu().contiguous().numpy()
            else:  # numpy ndarray, list, or tuple
                array = np.ascontiguousarray(np.asarray(value))
            digest = hashlib.sha256(memoryview(array)).digest()
            key = _array_content_key(value, digest)

        return key

    def _create_cache_key(self, wavelength: float | be.ndarray, **kwargs) -> tuple:
        """Creates a hashable cache key from wavelength and kwargs."""
        if be.is_array_like(wavelength):
            size = self._array_size(wavelength)
            if size is not None and size <= self._MAX_VALUE_KEY_ARRAY_SIZE:
                wavelength_key = (
                    "array-values",
                    tuple(np.ravel(be.to_numpy(wavelength))),
                    tuple(wavelength.shape),
                    str(wavelength.dtype),
                    str(getattr(wavelength, "device", None)),
                )
            else:
                wavelength_key = self._array_metadata_key(wavelength)
        else:
            wavelength_key = (type(wavelength), wavelength)
        return (wavelength_key, self._state_key(tuple(sorted(kwargs.items()))))

    def _cache_state(self) -> tuple | None:
        """Return a hashable optical-state token, or None to disable caching.

        An immutable model can return an empty tuple. Mutable models must include
        every parameter affecting n/k, and return None while parameters require
        gradients. `_state_key` fingerprints ordinary numerical data safely,
        including arrays mutated through aliases. Unknown custom state defaults
        to uncached evaluation; subclasses need not opt in to remain usable.
        Each concrete subclass must explicitly override this hook, including
        subclasses of built-in materials that might add optical state.
        """
        return None

    @classmethod
    def _state_key(cls, value) -> tuple | None:
        """Fingerprint numerical state; unsupported or trainable data disables reuse."""
        if cls._requires_grad(value):
            return None
        if isinstance(value, (tuple, list)):
            items = []
            for item in value:
                key = cls._state_key(item)
                if key is None:
                    return None
                items.append(key)
            return tuple(items)
        if isinstance(value, (str, int, float, bool, type(None))):
            return (type(value), value)
        if isinstance(value, np.ndarray) or (
            torch is not None and isinstance(value, torch.Tensor)
        ):
            if cls._array_size(value) <= cls._MAX_VALUE_KEY_ARRAY_SIZE:
                # Small live parameter arrays need a snapshot, not a uniformity
                # reduction or digest. Copy bytes so subsequent alias writes
                # cannot alter the stored state token.
                array = be.to_numpy(value)
                return _array_content_key(value, array.tobytes())
            return cls._array_metadata_key(value)
        return None

    @staticmethod
    def _backend_context() -> tuple:
        """Identify result type, precision, device and backend-created gradients."""
        backend = be.get_backend()
        device = be.get_device() if backend == "torch" else "cpu"
        gradients = be.grad_mode.requires_grad if backend == "torch" else False
        inference = torch.is_inference_mode_enabled() if backend == "torch" else False
        return backend, be.get_precision(), device, gradients, inference

    @staticmethod
    def _as_backend_array(value, *, preserve_dtype: bool = False):
        """Use live parameters in the active backend without replacing their storage.

        Typed parameters can opt out of conversion to the backend's default
        precision. Untyped values still use that default.
        """
        if be.get_backend() == "numpy" and hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if preserve_dtype and hasattr(value, "dtype"):
            return be.asarray(value, dtype=None)
        return be.asarray(value)

    @staticmethod
    def _requires_grad(value) -> bool:
        """Check if a value is a torch tensor that requires gradient."""
        return hasattr(value, "requires_grad") and value.requires_grad

    @staticmethod
    def _compute_grad_aware(calculate, wavelength, **kwargs):
        """Evaluate a property calculation with grad recording enabled.

        The first evaluation of a material property can happen inside a
        ``torch.no_grad()`` block -- the implicit-differentiation primal
        solves trace the system graph-free. Computed under ``no_grad``, a
        value derived from a *trainable* parameter reports
        ``requires_grad == False``, so the do-not-cache-differentiable-values
        check below would mistake it for a constant and cache it detached;
        every later grad-attached trace would then read the stale detached
        value and the parameter's gradient would silently collapse to zero.
        Evaluating under ``torch.enable_grad()`` (the same principle as the
        Forbes coefficient-cache fix) keeps a grad-connected result
        recognizable regardless of the caller's ambient grad context. Only
        this evaluation is recorded; the caller's ``no_grad`` scope still
        applies to everything downstream.
        """
        if torch is not None and be.get_backend() == "torch":
            with torch.enable_grad():
                return calculate(wavelength, **kwargs)
        return calculate(wavelength, **kwargs)

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

    @staticmethod
    def _is_uniform_key(cache_key: tuple) -> bool:
        """True when the key's wavelength part marks a constant array."""
        wavelength_key = cache_key[0]
        return (
            isinstance(wavelength_key, tuple)
            and len(wavelength_key) > 0
            and wavelength_key[0] == "array-uniform"
        )

    @staticmethod
    def _uniform_representative(wavelength):
        """A 1-element view of a constant wavelength array (same dtype/device)."""
        if torch is not None and isinstance(wavelength, torch.Tensor):
            return wavelength.reshape(-1)[:1]
        return np.asarray(wavelength).reshape(-1)[:1]

    @staticmethod
    def _broadcast_like(value, wavelength):
        """Expand a single-value result to the wavelength's shape as a view.

        The expansion allocates no memory (stride-0 view), so a property
        evaluated once per wavelength serves bundles of any size for free.
        """
        if torch is not None and isinstance(value, torch.Tensor):
            return value.reshape(1).expand(tuple(np.shape(wavelength)))
        return np.broadcast_to(np.asarray(value).reshape(1), np.shape(wavelength))

    def _evaluate_property(self, property_name: str, wavelength, **kwargs):
        """Evaluate n/k with one state-aware cache and fresh gradient graphs."""
        calculate = getattr(self, f"_calculate_{property_name}")
        if self._requires_grad(wavelength):
            return self._compute_grad_aware(calculate, wavelength, **kwargs)
        state = self._cache_state() if "_cache_state" in type(self).__dict__ else None
        kwargs_state = self._state_key(tuple(sorted(kwargs.items())))
        if state is None or kwargs_state is None:
            self._n_cache.clear()
            self._k_cache.clear()
            self._cache_context = None
            return self._compute_grad_aware(calculate, wavelength, **kwargs)

        context = (self._backend_context(), state)
        if context != self._cache_context:
            self._n_cache.clear()
            self._k_cache.clear()
            self._cache_context = context

        # Only non-trainable, tracked inputs reach cache lookup or reduction.
        # Equal values do not imply equal derivatives for each query element.
        cache = self._n_cache if property_name == "n" else self._k_cache
        cache_key = self._create_cache_key(wavelength, **kwargs)
        uniform = self._is_uniform_key(cache_key) and all(
            value is None or np.isscalar(value) for value in kwargs.values()
        )
        if cache_key not in cache:
            query = self._uniform_representative(wavelength) if uniform else wavelength
            result = self._compute_grad_aware(calculate, query, **kwargs)
            if self._requires_grad(result):
                return self._broadcast_like(result, wavelength) if uniform else result
            cache[cache_key] = self._detach_if_tensor(result)
        result = cache[cache_key]
        return self._broadcast_like(result, wavelength) if uniform else result

    def n(self, wavelength: float | be.ndarray, **kwargs) -> float | be.ndarray:
        """Calculates the refractive index at a given wavelength with caching.

        Args:
            wavelength (float | be.ndarray): The wavelength(s) of light in microns.
                Can be a float, numpy array, or torch tensor.
            **kwargs: Additional keyword arguments for calculation (e.g., temperature).

        Returns:
            float | be.ndarray: The refractive index at the given wavelength(s).
        """
        return self._evaluate_property("n", wavelength, **kwargs)

    def k(self, wavelength: float | be.ndarray, **kwargs) -> float | be.ndarray:
        """Calculates the extinction coefficient at a given wavelength with caching.

        Args:
            wavelength (float | be.ndarray): The wavelength(s) of light in microns.
                Can be a float, numpy array, or torch tensor.
            **kwargs: Additional keyword arguments for calculation.

        Returns:
            float | be.ndarray: The extinction coefficient at the given wavelength(s).
        """
        return self._evaluate_property("k", wavelength, **kwargs)

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
            raise ValueError(
                f"Unknown material type {material_type!r} in the material data."
                f"{options_hint(str(material_type), cls._registry)}"
            )

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
