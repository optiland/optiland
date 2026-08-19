"""
Provides the phase interaction model.
"""

from __future__ import annotations

import typing

from optiland import backend as be
from optiland.interactions.base import BaseInteractionModel
from optiland.phase.base import BasePhaseProfile

if typing.TYPE_CHECKING:
    from optiland.raytrace.rays import ParaxialRays, RealRays


class PhaseInteractionModel(BaseInteractionModel):
    """An interaction model for surfaces with an arbitrary phase profile.

    This model implements the Generalized Snell's Law, where the change in a
    ray's transverse wavevector is determined by the gradient of a phase
    profile defined on the surface. It is a "thin sheet" model and must be
    used with a `Plane` geometry.

    Args:
        parent_surface (Surface): The surface to which this interaction model is
            attached.
        phase_profile (BasePhaseProfile): The phase profile strategy object that
            defines the phase and its gradient.
    """

    interaction_type = "phase"

    def __init__(self, parent_surface, phase_profile, is_reflective, **kwargs):
        self.phase_profile = phase_profile
        self._parent_surface = None
        super().__init__(parent_surface, is_reflective=is_reflective, **kwargs)

    @property
    def parent_surface(self):
        return self._parent_surface

    @parent_surface.setter
    def parent_surface(self, value):
        self._parent_surface = value
        self.phase_profile.parent_surface = value

    def interact_real_rays(self, rays: RealRays) -> RealRays:
        if self.parent_surface is None:
            raise RuntimeError("Parent surface not set for PhaseInteractionModel.")

        # Get incident state
        x, y = rays.x, rays.y
        l_i, m_i, n_i = rays.L, rays.M, rays.N
        rays.L0, rays.M0, rays.N0 = l_i, m_i, n_i
        n1 = self.parent_surface.material_pre.n(rays.w)
        n2 = self.parent_surface.material_post.n(rays.w)
        if self.is_reflective:
            n2 = n1
        k0 = 2 * be.pi / (rays.w * 1e-3)  # convert to mm

        # 1. Get local surface normal (N)
        nx, ny, nz = self.parent_surface.geometry.surface_normal(rays)

        # 2. Get incident wavevector (k_in)
        k_ix = n1 * k0 * l_i
        k_iy = n1 * k0 * m_i
        k_iz = n1 * k0 * n_i

        # 3. Get phase and ambient gradient (grad(f))
        phase_val = self.phase_profile.get_phase(x, y, rays.w)
        phi_x, phi_y, phi_z = self.phase_profile.get_gradient(x, y, rays.w)
        grad_f_x = phi_x
        grad_f_y = phi_y
        grad_f_z = phi_z

        # 4. Compute surface gradient (G = grad(f) - (grad(f)·N)N)
        grad_f_dot_N = grad_f_x * nx + grad_f_y * ny + grad_f_z * nz
        G_x = grad_f_x - grad_f_dot_N * nx
        G_y = grad_f_y - grad_f_dot_N * ny
        G_z = grad_f_z - grad_f_dot_N * nz

        # 5. Compute incident tangential component (k_in,‖ = k_in - (k_in·N)N)
        k_in_dot_N = k_ix * nx + k_iy * ny + k_iz * nz
        k_in_par_x = k_ix - k_in_dot_N * nx
        k_in_par_y = k_iy - k_in_dot_N * ny
        k_in_par_z = k_iz - k_in_dot_N * nz

        # 6. Compute outgoing tangential component (k_out,‖ = k_in,‖ + G)
        k_out_par_x = k_in_par_x + G_x
        k_out_par_y = k_in_par_y + G_y
        k_out_par_z = k_in_par_z + G_z

        # 7. Compute outgoing normal component (alpha)
        k_out_par_mag_sq = k_out_par_x**2 + k_out_par_y**2 + k_out_par_z**2
        k_out_mag_sq = (n2 * k0) ** 2
        R_sq = k_out_mag_sq - k_out_par_mag_sq  # (alpha)^2

        # 8. Handle TIR/Evanescence
        tir_mask = R_sq < 0.0
        rays.clip(tir_mask)
        R_sq = be.maximum(0.0, R_sq)
        alpha_mag = be.sqrt(R_sq)

        # 9. Choose sign for alpha (Normal component)
        # Refraction: points along +N
        # Reflection: points along -N
        alpha_sign = -1.0 if self.is_reflective else 1.0
        alpha = alpha_sign * alpha_mag

        # 10. Build full outgoing wavevector (k_out = k_out,‖ + alpha*N)
        k_out_x = k_out_par_x + alpha * nx
        k_out_y = k_out_par_y + alpha * ny
        k_out_z = k_out_par_z + alpha * nz

        # 11. Get new direction cosines
        k_out_mag = be.sqrt(k_out_x**2 + k_out_y**2 + k_out_z**2)
        l_o = k_out_x / k_out_mag
        m_o = k_out_y / k_out_mag
        n_o = k_out_z / k_out_mag

        # Update ray
        rays.L, rays.M, rays.N = l_o, m_o, n_o

        # Update OPD
        opd_shift = -phase_val / k0
        rays.opd = rays.opd + opd_shift

        # Apply coating/BSDF
        rays = self._apply_coating_and_bsdf(rays, nx, ny, nz)

        # Apply phase profile efficiency
        rays.i = rays.i * self.phase_profile.efficiency

        return rays

    def flip(self):
        """Flips the interaction model.

        For a phase profile defined on a plane, this operation does nothing.
        """
        pass

    def interact_paraxial_rays(self, rays: ParaxialRays) -> ParaxialRays:
        """Applies the phase gradient interaction to paraxial rays.

        Args:
            rays: The paraxial rays to interact with the surface.

        Returns:
            The interacted paraxial rays.
        """
        # Get incident state
        n1 = self.parent_surface.material_pre.n(rays.w)
        n2 = self.parent_surface.material_post.n(rays.w)
        k0 = 2 * be.pi / (rays.w * 1e-3)  # convert to mm
        y = rays.y

        # Get paraxial gradient from the strategy
        paraxial_gradient = self.phase_profile.get_paraxial_gradient(y)

        # Apply geometric + gradient deflection
        grad_deflection = paraxial_gradient / k0

        if self.is_reflective:
            n = n1
            power = (
                -2 * n / self.parent_surface.geometry.radius
            )  # Will be zero for Plane
            u_geom = rays.u - y * power / n
            rays.u = u_geom + grad_deflection / n
        else:
            # The sign of grad_deflection is flipped to match the convention
            # in the legacy DiffractiveInteractionModel.
            power = (n2 - n1) / self.parent_surface.geometry.radius
            u_geom = (n1 / n2) * rays.u - y * power / n2
            rays.u = u_geom - grad_deflection / n2

        return rays

    def to_dict(self) -> dict:
        """Serializes the phase profile to a dictionary.

        Returns:
            A dictionary representation of the interaction model.
        """
        data = super().to_dict()
        data.update({"phase_profile": self.phase_profile.to_dict()})
        return data

    @classmethod
    def _deserialize_init_data(cls, data):
        init_data = super()._deserialize_init_data(data)
        init_data["phase_profile"] = BasePhaseProfile.from_dict(
            init_data["phase_profile"]
        )
        return init_data
