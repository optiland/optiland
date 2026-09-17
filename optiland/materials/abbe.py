"""Abbe Material

This module defines a material based on the refractive index and Abbe number.
It provides support for both d-line (587.56 nm) and e-line (546.07 nm)
definitions.

Models:
1.  "polynomial": A polynomial fit to glass data (Legacy, D-line only).
2.  "buchdahl": A Buchdahl 3-term model with LASSO-derived coefficients.

Kramer Harrison, 2024
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from importlib import resources

import numpy as np

import optiland.backend as be
from optiland.materials.base import BaseMaterial
from optiland.materials.buchdahl import buchdahl_coordinate, evaluate_buchdahl


class AbbeModel(ABC):
    """Abstract base class for Abbe number based material models."""

    @abstractmethod
    def predict_n(self, wavelength: float | be.ndarray) -> float | be.ndarray:
        """Predicts the refractive index at a given wavelength."""
        pass

    @abstractmethod
    def predict_k(self, wavelength: float | be.ndarray) -> float | be.ndarray:
        """Predicts the extinction coefficient at a given wavelength."""
        pass


class AbbePolynomialModel(AbbeModel):
    """Legacy polynomial model for Abbe materials (d-line).

    This model uses a polynomial fit to glass data from the Schott catalog.
    It corresponds to the original implementation of AbbeMaterial.
    """

    def __init__(self, index: float, abbe: float):
        self.index = be.array([index])
        self.abbe = be.array([abbe])
        coefficients_file = resources.files("optiland.database").joinpath(
            "glass_model_coefficients.npy"
        )
        self._fit_coefficients = np.load(str(coefficients_file))
        self._p = self._get_coefficients()

    def _cache_state(self) -> tuple | None:
        """Track the parameters and fixed polynomial fit used by the wrapper."""
        return BaseMaterial._state_key(
            ("polynomial", self.index, self.abbe, self._fit_coefficients)
        )

    def predict_n(self, wavelength: float | be.ndarray) -> float | be.ndarray:
        wavelength = be.array(wavelength)
        if be.any(wavelength < 0.380) or be.any(wavelength > 0.750):
            # This legacy check is preserved
            raise ValueError("Wavelength out of range for this model.")
        # Recompute the small coefficient vector from live parameters. A stored
        # derived graph cannot be reused after backward or a parameter update.
        self._p = self._get_coefficients()
        return be.atleast_1d(be.polyval(self._p, wavelength))

    def predict_k(self, wavelength: float | be.ndarray) -> float | be.ndarray:
        return be.zeros_like(0)

    def _get_coefficients(self):
        # Polynomial fit to the refractive index data
        index = BaseMaterial._as_backend_array(self.index)
        abbe = BaseMaterial._as_backend_array(self.abbe)
        X_poly = be.ravel(
            be.array(
                [
                    index,
                    abbe,
                    index**2,
                    abbe**2,
                    index**3,
                    abbe**3,
                ]
            )
        )

        coefficients = BaseMaterial._as_backend_array(self._fit_coefficients)
        return be.matmul(X_poly, coefficients)


class BuchdahlModel(AbbeModel):
    """Base class for Buchdahl 3-term models."""

    ALPHA = 2.5

    def __init__(self, index, abbe):
        self.index = be.array([index])
        self.abbe = be.array([abbe])
        self.v1, self.v2, self.v3 = self._calculate_buchdahl_coefficients()

    def _cache_state(self) -> tuple | None:
        """Track reference convention, live inputs and configurable fitted constants."""
        return BaseMaterial._state_key(
            (
                "buchdahl",
                self.index,
                self.abbe,
                self.WAVE_REF,
                self.ALPHA,
                getattr(self, "V1_COEFFS", None),
                getattr(self, "V2_COEFFS", None),
                getattr(self, "V3_COEFFS", None),
            )
        )

    @property
    @abstractmethod
    def WAVE_REF(self) -> float:
        """Reference wavelength in microns."""
        pass

    @abstractmethod
    def _calculate_buchdahl_coefficients(self) -> tuple[float, float, float]:
        """Calculates v1, v2, v3 based on index and abbe."""
        pass

    def predict_n(self, wavelength: float | be.ndarray) -> float | be.ndarray:
        self.v1, self.v2, self.v3 = self._calculate_buchdahl_coefficients()
        omega = buchdahl_coordinate(wavelength, self.WAVE_REF, self.ALPHA)
        n_pred = evaluate_buchdahl(
            BaseMaterial._as_backend_array(self.index),
            (self.v1, self.v2, self.v3),
            omega,
        )
        return be.atleast_1d(n_pred)

    def predict_k(self, wavelength: float | be.ndarray) -> float | be.ndarray:
        return be.zeros_like(0)


class BuchdahlDModel(BuchdahlModel):
    """Buchdahl model for d-line (587.56 nm) materials."""

    WAVE_REF = 0.5875618

    # Coefficients for v1 prediction
    V1_COEFFS = [
        0.004160,  # Intercept
        4.462559,  # 1/V
        2.326660,  # 1/V^2
        0.002330,  # n
        -0.003697,  # n^2
        -4.697604,  # n/V
    ]

    # Coefficients for v2 prediction
    V2_COEFFS = [
        0.066434,  # Intercept
        -7.636396,  # 1/V
        12.597434,  # 1/V^2
        -0.037014,  # n^2
        5.551013,  # n/V
    ]

    # Coefficients for v3 prediction
    V3_COEFFS = [
        -0.032218,  # Intercept
        2.230357,  # 1/V
        -103.318994,  # 1/V^2
        -0.009654,  # n^2
        1.934983,  # n/V
    ]

    def _calculate_buchdahl_coefficients(self):
        nd = BaseMaterial._as_backend_array(self.index)
        vd = BaseMaterial._as_backend_array(self.abbe)
        inv_v = 1.0 / vd
        inv_v2 = 1.0 / (vd**2)
        nd_sq = nd**2
        nd_div_v = nd / vd

        # Calculate v1
        # Terms: [1, 1/V, 1/V^2, n, n^2, n/V]
        c = self.V1_COEFFS
        v1 = (
            c[0]
            + c[1] * inv_v
            + c[2] * inv_v2
            + c[3] * nd
            + c[4] * nd_sq
            + c[5] * nd_div_v
        )

        # Calculate v2
        # Terms: [1, 1/V, 1/V^2, n^2, n/V]
        c = self.V2_COEFFS
        v2 = c[0] + c[1] * inv_v + c[2] * inv_v2 + c[3] * nd_sq + c[4] * nd_div_v

        # Calculate v3
        # Terms: [1, 1/V, 1/V^2, n^2, n/V]
        c = self.V3_COEFFS
        v3 = c[0] + c[1] * inv_v + c[2] * inv_v2 + c[3] * nd_sq + c[4] * nd_div_v

        return v1, v2, v3


class BuchdahlEModel(BuchdahlModel):
    """Buchdahl model for e-line (546.07 nm) materials."""

    WAVE_REF = 0.546074

    def _calculate_buchdahl_coefficients(self):
        ne = BaseMaterial._as_backend_array(self.index)
        ve = BaseMaterial._as_backend_array(self.abbe)

        inv_v = 1.0 / ve
        inv_v2 = 1.0 / (ve**2)
        n_sq = ne**2
        n_div_v = ne / ve

        # Derived via Lasso regression
        v1 = (
            -0.01271580
            + 5.86039368 * inv_v
            + 0.00000000 * inv_v2
            - 0.00840567 * n_sq
            - 6.04120358 * n_div_v
        )

        v2 = (
            -0.11714561
            - 19.45035516 * inv_v
            + 0.00000000 * inv_v2
            - 0.18747797 * n_sq
            + 14.33541100 * n_div_v
        )

        v3 = (
            0.00000000
            + 18.43536735 * inv_v
            - 241.00526954 * inv_v2
            + 0.10881050 * n_sq
            - 4.93439893 * n_div_v
        )

        return v1, v2, v3


class _AbbeMaterialParameters:
    """Expose model-owned parameters identically through both public wrappers."""

    model: AbbeModel

    @property
    def display_name(self) -> str:
        """Show the reference index and Abbe number with their line convention."""
        label = f"{self.index.item():.4f}, {self.abbe.item():.2f}"
        return label + (" (ne, Ve)" if isinstance(self, AbbeMaterialE) else "")

    def spectral_range(self, property_name: str = "n") -> tuple[float, float] | None:
        """The polynomial fit's known range; Buchdahl has no declared limits."""
        super().spectral_range(property_name)
        return (
            (0.38, 0.75)
            if property_name == "n" and isinstance(self.model, AbbePolynomialModel)
            else None
        )

    @property
    def index(self):
        """The model's live reference-line refractive index."""
        return self.model.index

    @index.setter
    def index(self, value):
        self.model.index = value

    @property
    def abbe(self):
        """The model's live Abbe number in its reference-line convention."""
        return self.model.abbe

    @abbe.setter
    def abbe(self, value):
        self.model.abbe = value

    def _model_cache_state(self) -> tuple | None:
        if type(self.model) not in (
            AbbePolynomialModel,
            BuchdahlDModel,
            BuchdahlEModel,
        ):
            return None
        return self.model._cache_state()

    def _calculate_n(self, wavelength, **kwargs):
        """Evaluate the selected model from its authoritative parameters."""
        return self.model.predict_n(wavelength)

    def _calculate_k(self, wavelength, **kwargs):
        """Evaluate extinction through the existing Abbe model contract."""
        return self.model.predict_k(wavelength)


class AbbeMaterial(_AbbeMaterialParameters, BaseMaterial):
    """Represents a material based on the refractive index and Abbe number.

    This class serves as a wrapper around specific model implementations.
    Currently supported models:
    - 'polynomial': The legacy polynomial fit (Default, deprecated).
    - 'buchdahl': The new Buchdahl 3-term model (Recommended).

    Args:
        n (float): The refractive index of the material at 587.56 nm (n_d).
        abbe (float): The Abbe number of the material (V_d).
        model (str, optional): The model to use. Defaults to "polynomial".
            Valid options are "polynomial" and "buchdahl".

    Attributes:
        index (float): The refractive index of the material at 587.56 nm.
        abbe (float): The Abbe number of the material.
        model_name (str): The name of the model being used.
        model (AbbeModel): The underlying model instance.

    """

    def __init__(self, n, abbe, model=None):
        super().__init__()

        if model is None:
            warnings.warn(
                "The default model for AbbeMaterial will change from 'polynomial' "
                "to 'buchdahl' in v0.7.0. The 'buchdahl' model offers improved "
                "accuracy. To silence this warning, specify `model='polynomial'` "
                "explicitly if you intended to use the legacy model, or switch to "
                "`model='buchdahl'`.",
                FutureWarning,
                stacklevel=2,
            )
            model = "polynomial"

        self.model_name = model

        if model == "polynomial":
            self.model = AbbePolynomialModel(n, abbe)
        elif model == "buchdahl":
            self.model = BuchdahlDModel(n, abbe)
        else:
            raise ValueError(
                f"Unknown model: {model}. Valid options: 'polynomial', 'buchdahl'"
            )

    def _cache_state(self) -> tuple | None:
        """Track the selected d-line model's authoritative parameters."""
        return self._model_cache_state()

    def to_dict(self):
        """Returns a dictionary representation of the material."""
        material_dict = super().to_dict()
        material_dict.update(
            {
                "index": float(self.index.item()),
                "abbe": float(self.abbe.item()),
                "model": self.model_name,
            }
        )
        return material_dict

    @classmethod
    def from_dict(cls, data):
        """Creates a material from a dictionary representation."""
        required_keys = ["index", "abbe"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        model = data.get("model")
        return cls(data["index"], data["abbe"], model=model)


class AbbeMaterialE(_AbbeMaterialParameters, BaseMaterial):
    """Represents a material based on the refractive index and Abbe number at e-line.

    This class uses a Buchdahl 3-term model fitted to e-line (546.07 nm) data.

    Args:
        n (float): The refractive index of the material at 546.07 nm (n_e).
        abbe (float): The Abbe number of the material (V_e).

    Attributes:
        index (float): The refractive index of the material at 546.07 nm.
        abbe (float): The Abbe number of the material (V_e).
        model (BuchdahlEModel): The underlying model instance.

    """

    def __init__(self, n, abbe):
        super().__init__()
        self.model = BuchdahlEModel(n, abbe)

    def _cache_state(self) -> tuple | None:
        """Track the e-line model's authoritative parameters."""
        return self._model_cache_state()

    def to_dict(self):
        """Returns a dictionary representation of the material."""
        material_dict = super().to_dict()
        material_dict.update(
            {
                "index": float(self.index.item()),
                "abbe": float(self.abbe.item()),
            }
        )
        return material_dict

    @classmethod
    def from_dict(cls, data):
        """Creates a material from a dictionary representation."""
        required_keys = ["index", "abbe"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        return cls(data["index"], data["abbe"])
