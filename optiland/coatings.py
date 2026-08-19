"""Coatings Module

The coatings module contains classes for modeling optical coatings.

Kramer Harrison, 2024
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.jones import (
    BaseJones,
    JonesFresnel,
    JonesLinearPolarizer,
    JonesLinearRetarder,
)
from optiland.materials import BaseMaterial
from optiland.thin_film import ThinFilmStack

if TYPE_CHECKING:
    from optiland.rays import RealRays


class BaseCoating(ABC):
    """Base class for coatings.

    This class defines the basic structure and behavior of a coating.

    Methods:
        interact: Performs an interaction with the coating.
        reflect: Abstract method to handle reflection interaction with the coating.
        transmit: Abstract method to handle transmission interaction with the coating.

    """

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseCoating._registry[cls.__name__] = cls

    def interact(
        self,
        rays: RealRays,
        reflect: bool = False,
        nx: be.ndarray = None,
        ny: be.ndarray = None,
        nz: be.ndarray = None,
    ) -> RealRays:
        """Performs an interaction with the coating.

        Args:
            rays (RealRays): The rays incident on the coating.
            reflect (bool, optional): Flag indicating whether to perform
                reflection (True) or transmission (False). Defaults to False.
            nx (be.ndarray, optional): The x-component of the surface normal vectors.
            ny (be.ndarray, optional): The y-component of the surface normal vectors.
            nz (be.ndarray, optional): The z-component of the surface normal vectors.

        Returns:
            rays (RealRays): The rays after the interaction.

        """
        if reflect:
            return self.reflect(rays, nx, ny, nz)
        return self.transmit(rays, nx, ny, nz)

    def _compute_aoi(
        self, rays: RealRays, nx: be.ndarray, ny: be.ndarray, nz: be.ndarray
    ) -> be.ndarray:
        """Computes the angle of incidence for the given rays and surface normals.

        Args:
            rays (RealRays): The incident rays.
            nx (be.ndarray): The x-component of the surface normal vectors at each ray's
                intersection point.
            ny (be.ndarray): The y-component of the surface normal vectors at each ray's
                intersection point.
            nz (be.ndarray): The z-component of the surface normal vectors at each ray's
                intersection point.

        Returns:
            be.ndarray: The angle of incidence for each ray.

        """
        dot = be.abs(nx * rays.L0 + ny * rays.M0 + nz * rays.N0)
        dot = be.clip(dot, -1, 1)  # required due to numerical precision
        return be.arccos(dot)

    @abstractmethod
    def reflect(
        self,
        rays: RealRays,
        nx: be.ndarray = None,
        ny: be.ndarray = None,
        nz: be.ndarray = None,
    ) -> RealRays:
        """Abstract method to handle reflection interaction.

        Args:
            rays (RealRays): The rays incident on the coating.
            nx (be.ndarray, optional): The x-component of the surface normal vectors.
            ny (be.ndarray, optional): The y-component of the surface normal vectors.
            nz (be.ndarray, optional): The z-component of the surface normal vectors.

        Returns:
            RealRays: The rays after reflection.

        """
        # pragma: no cover

    @abstractmethod
    def transmit(
        self,
        rays: RealRays,
        nx: be.ndarray = None,
        ny: be.ndarray = None,
        nz: be.ndarray = None,
    ) -> RealRays:
        """Abstract method to handle transmission interaction.

        Args:
            rays (RealRays): The rays incident on the coating.
            nx (be.ndarray, optional): The x-component of the surface normal vectors.
            ny (be.ndarray, optional): The y-component of the surface normal vectors.
            nz (be.ndarray, optional): The z-component of the surface normal vectors.

        Returns:
            RealRays: The rays after transmission.

        """
        # pragma: no cover

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover
        """Converts the coating to a dictionary.

        Returns:
            dict: The dictionary representation of the coating.

        """
        return {
            "type": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseCoating:
        """Creates a coating from a dictionary.

        Args:
            data (dict): The dictionary representation of the coating.

        Returns:
            BaseCoating: The coating created from the dictionary.

        """
        coating_type = data["type"]
        return cls._registry[coating_type].from_dict(data)


class SimpleCoating(BaseCoating):
    """A simple coating class that represents a coating with given transmittance
    and reflectance.

    Args:
        transmittance (float): The transmittance of the coating.
        reflectance (float, optional): The reflectance of the coating.
            Defaults to 0.

    Attributes:
        transmittance (float): The transmittance of the coating.
        reflectance (float): The reflectance of the coating.
        absorptance (float): The absorptance of the coating, calculated
            as 1 - reflectance - transmittance.

    Methods:
        reflect(rays: RealRays, nx: be.ndarray = None, ny: be.ndarray = None,
            nz: be.ndarray = None) -> RealRays: Reflects the rays based on the
            reflectance of the coating.
        transmit(rays: RealRays, nx: be.ndarray = None, ny: be.ndarray = None,
            nz: be.ndarray = None) -> RealRays: Transmits the rays based on the
            transmittance of the coating.

    """

    def __init__(self, transmittance: float, reflectance: float = 0):
        self.transmittance = transmittance
        self.reflectance = reflectance
        self.absorptance = 1 - reflectance - transmittance

    def reflect(
        self,
        rays: RealRays,
        nx: be.ndarray = None,
        ny: be.ndarray = None,
        nz: be.ndarray = None,
    ) -> RealRays:
        """Reflects the rays based on the reflectance of the coating.

        Args:
            rays (RealRays): The rays incident on the coating.
            nx (be.ndarray, optional): The x-component of the surface normal vectors.
            ny (be.ndarray, optional): The y-component of the surface normal vectors.
            nz (be.ndarray, optional): The z-component of the surface normal vectors.

        Returns:
            RealRays: The rays after reflection.

        """
        rays.i = rays.i * self.reflectance
        return rays

    def transmit(
        self,
        rays: RealRays,
        nx: be.ndarray = None,
        ny: be.ndarray = None,
        nz: be.ndarray = None,
    ) -> RealRays:
        """Transmits the rays through the coating by multiplying their intensity
        with the transmittance.

        Args:
            rays (RealRays): The rays incident on the coating.
            nx (be.ndarray, optional): The x-component of the surface normal vectors.
            ny (be.ndarray, optional): The y-component of the surface normal vectors.
            nz (be.ndarray, optional): The z-component of the surface normal vectors.

        Returns:
            RealRays: The rays after transmission.

        """
        rays.i = rays.i * self.transmittance
        return rays

    def to_dict(self) -> dict[str, Any]:
        """Converts the coating to a dictionary.

        Returns:
            dict: The dictionary representation of the coating.

        """
        return {
            "type": self.__class__.__name__,
            "transmittance": self.transmittance,
            "reflectance": self.reflectance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimpleCoating:
        """Creates a coating from a dictionary.

        Args:
            data (dict): The dictionary representation of the coating.

        Returns:
            BaseCoating: The coating created from the dictionary.

        """
        return cls(data["transmittance"], data["reflectance"])


class BaseCoatingPolarized(BaseCoating, ABC):
    """A base class for polarized coatings.

    This class inherits from the `BaseCoating` class and the `ABC`
    (Abstract Base Class) module. Any subclass must implement the `jones`
    property to provide the Jones matrix model for the coating.

    Methods:
        reflect(rays, nx, ny, nz): Reflects the rays off the coating.
        transmit(rays, nx, ny, nz): Transmits the rays through the coating.

    """

    @property
    @abstractmethod
    def jones(self) -> BaseJones:
        """The Jones matrix model associated with the coating."""
        pass  # pragma: no cover

    def reflect(
        self,
        rays: RealRays,
        nx: be.ndarray = None,
        ny: be.ndarray = None,
        nz: be.ndarray = None,
    ) -> RealRays:
        """Reflects the rays off the coating.

        Args:
            rays (RealRays): The rays to be reflected.
            nx (be.ndarray, optional): The x-component of the surface normal vector.
            ny (be.ndarray, optional): The y-component of the surface normal vector.
            nz (be.ndarray, optional): The z-component of the surface normal vector.

        Returns:
            RealRays: The updated rays after reflection.

        """
        aoi = self._compute_aoi(rays, nx, ny, nz)
        jones = self.jones.calculate_matrix(rays, reflect=True, aoi=aoi)
        rays.update(jones)
        return rays

    def transmit(
        self,
        rays: RealRays,
        nx: be.ndarray = None,
        ny: be.ndarray = None,
        nz: be.ndarray = None,
    ) -> RealRays:
        """Transmits the rays through the coating.

        Args:
            rays (RealRays): The rays to be transmitted.
            nx (be.ndarray, optional): The x-component of the surface normal vector.
            ny (be.ndarray, optional): The y-component of the surface normal vector.
            nz (be.ndarray, optional): The z-component of the surface normal vector.

        Returns:
            RealRays: The updated rays after transmission through a surface.

        """
        aoi = self._compute_aoi(rays, nx, ny, nz)
        jones = self.jones.calculate_matrix(rays, reflect=False, aoi=aoi)
        rays.update(jones)
        return rays

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover
        """Converts the coating to a dictionary.

        Returns:
            dict: The dictionary representation of the coating.

        """
        return {
            "type": self.__class__.__name__,
            "material_pre": self.material_pre.to_dict(),
            "material_post": self.material_post.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> BaseCoatingPolarized:  # pragma: no cover
        """Creates a coating from a dictionary.

        Args:
            data (dict): The dictionary representation of the coating.

        Returns:
            BaseCoating: The coating created from the dictionary.

        """
        return cls(data["material_pre"], data["material_post"])


class FresnelCoating(BaseCoatingPolarized):
    """Represents a Fresnel coating for polarized light.

    This class inherits from the BaseCoatingPolarized class and provides
    interaction functionality for polarized light with uncoated surfaces.
    In general, this updates ray intensities based on the Fresnel equations
    on a surface.

    Attributes:
        material_pre (str): The material before the coating.
        material_post (str): The material after the coating.
        jones (JonesFresnel): The JonesFresnel object, which calculates the
            Jones matrices for given ray properties.

    """

    def __init__(self, material_pre: BaseMaterial, material_post: BaseMaterial):
        self.material_pre = material_pre
        self.material_post = material_post

        self._jones = JonesFresnel(material_pre, material_post)

    @property
    def jones(self) -> JonesFresnel:
        return self._jones

    def to_dict(self) -> dict[str, Any]:
        """Converts the coating to a dictionary.

        Returns:
            dict: The dictionary representation of the coating.

        """
        return {
            "type": self.__class__.__name__,
            "material_pre": self.material_pre.to_dict(),
            "material_post": self.material_post.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FresnelCoating:
        """Creates a coating from a dictionary.

        Args:
            data (dict): The dictionary representation of the coating.

        Returns:
            BaseCoating: The coating created from the dictionary.

        """
        return cls(
            BaseMaterial.from_dict(data["material_pre"]),
            BaseMaterial.from_dict(data["material_post"]),
        )


class PolarizerCoating(BaseCoatingPolarized):
    """Represents a linear polarizer coating.

    Args:
        axis (tuple | list | be.ndarray): A 3D vector representing the transmission
            axis in global coordinates. Defaults to [1.0, 0.0, 0.0] (horizontal).
    """

    def __init__(
        self,
        axis: tuple[float, float, float] | list[float] | be.ndarray = (1.0, 0.0, 0.0),
    ):
        self.axis = axis
        self._jones = JonesLinearPolarizer(axis)

    @property
    def jones(self) -> JonesLinearPolarizer:
        return self._jones

    def to_dict(self) -> dict[str, Any]:
        """Converts the coating to a dictionary."""
        return {
            "type": self.__class__.__name__,
            "axis": list(self.axis) if not isinstance(self.axis, list) else self.axis,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolarizerCoating:
        """Creates a coating from a dictionary."""
        return cls(axis=data.get("axis", (1.0, 0.0, 0.0)))


class RetarderCoating(BaseCoatingPolarized):
    """Represents a linear retarder coating.

    Args:
        retardance (float): The retardance of the coating in radians.
        axis (tuple | list | be.ndarray): A 3D vector representing the fast axis
            in global coordinates. Defaults to [1.0, 0.0, 0.0] (horizontal).
    """

    def __init__(
        self,
        retardance: float,
        axis: tuple[float, float, float] | list[float] | be.ndarray = (1.0, 0.0, 0.0),
    ):
        self.retardance = retardance
        self.axis = axis
        self._jones = JonesLinearRetarder(retardance, axis)

    @property
    def jones(self) -> JonesLinearRetarder:
        return self._jones

    def to_dict(self) -> dict[str, Any]:
        """Converts the coating to a dictionary."""
        return {
            "type": self.__class__.__name__,
            "retardance": self.retardance,
            "axis": list(self.axis) if not isinstance(self.axis, list) else self.axis,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetarderCoating:
        """Creates a coating from a dictionary."""
        return cls(
            retardance=data["retardance"], axis=data.get("axis", (1.0, 0.0, 0.0))
        )


class JonesThinFilm(BaseJones):
    """Jones matrix generator for a thin-film stack.

    Builds diagonal Jones matrices in the s/p basis using thin-film r/t
    amplitude coefficients. Reflect or transmit selection mirrors JonesFresnel.

    Args:
        stack: ThinFilmStack configured with incident/substrate and layers.
        wavelength_nm: Optional wavelength override (nm); if None uses rays.w (µm)
        converted.
        aoi_override_rad: Optional AOI override (radians); if None uses computed AOI.
    """

    def __init__(self, stack: ThinFilmStack):
        self.stack = stack

    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ) -> be.ndarray:
        # wavelengths: rays.w is in microns in Optiland
        wl_um = be.atleast_1d(rays.w)
        th = be.atleast_1d(aoi if aoi is not None else be.zeros_like(rays.w))

        # Compute s/p amplitudes per-ray; expect broadcasting over (N,)
        r_s, t_s, _, _ = self._coeffs_amp(wl_um, th, pol="s", reflect=reflect)
        r_p, t_p, _, _ = self._coeffs_amp(wl_um, th, pol="p", reflect=reflect)

        z = be.zeros_like(r_s)
        o = be.ones_like(r_s)

        if reflect:
            col0 = be.stack([r_s, z, z], axis=-1)
            col1 = be.stack([z, -r_p, z], axis=-1)
            col2 = be.stack([z, z, -o], axis=-1)
        else:
            col0 = be.stack([t_s, z, z], axis=-1)
            col1 = be.stack([z, t_p, z], axis=-1)
            col2 = be.stack([z, z, o], axis=-1)

        jones = be.stack([col0, col1, col2], axis=-2)
        return jones

    def _coeffs_amp(
        self, wl_um: be.ndarray, th_rad: be.ndarray, pol: str, reflect: bool
    ) -> tuple[be.ndarray, be.ndarray, be.ndarray, be.ndarray]:
        # Use internal helpers returning amplitudes from the stack TMM
        # We compute on per-ray vectors so shapes are (N,)
        out = self.stack.compute_rtRTA_elementwise(wl_um, th_rad, pol)
        r, t = out["r"], out["t"]
        R, T = out["R"], out["T"]
        return r, t, R, T


class ThinFilmCoating(BaseCoatingPolarized):
    """Polarized coating that applies a thin-film stack to rays.

    This class mirrors FresnelCoating but uses a ThinFilmStack to compute the
    s/p amplitude coefficients and builds a Jones matrix per ray via JonesThinFilm.

    Args:
        material_pre: Material before the stack (incident medium of the stack).
        material_post: Material after the stack (substrate of the stack).
        layers: Optional list of (material, thickness_nm, name) to build the stack.
    """

    def __init__(
        self,
        material_pre: BaseMaterial,
        material_post: BaseMaterial,
        layers: list[tuple[BaseMaterial, float, str | None]] | None = None,
    ):
        self.material_pre = material_pre
        self.material_post = material_post
        self.stack = ThinFilmStack(material_pre, material_post)
        if layers:
            for mat, thickness_nm, name in layers:
                self.stack.add_layer_nm(mat, thickness_nm, name)
        self._jones = JonesThinFilm(self.stack)

    @property
    def jones(self) -> JonesThinFilm:
        """The Jones matrix model associated with the thin-film coating."""
        return self._jones

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover
        return {
            "type": self.__class__.__name__,
            "material_pre": self.material_pre.to_dict(),
            "material_post": self.material_post.to_dict(),
            "layers": [
                {
                    "material": layer.material.to_dict(),
                    "thickness_nm": layer.thickness_um * 1000.0,
                    "name": layer.name,
                }
                for layer in self.stack.layers
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThinFilmCoating:  # pragma: no cover
        mats = []
        for d in data.get("layers", []):
            mats.append(
                (
                    BaseMaterial.from_dict(d["material"]),
                    d["thickness_nm"],
                    d.get("name"),
                )
            )
        return cls(
            BaseMaterial.from_dict(data["material_pre"]),
            BaseMaterial.from_dict(data["material_post"]),
            mats,
        )
