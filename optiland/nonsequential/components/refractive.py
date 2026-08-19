"""Refractive component for Non-Sequential Raytracing.

Transmits and reflects (lenses, prisms, windows). Uses detached-sample /
attached-weight Fresnel splitting for differentiable Monte Carlo.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential.components.base import BaseComponent
from optiland.nonsequential.components.coating_support import (
    reject_polarized_coating,
)
from optiland.nonsequential.materials.nsq_material import medium_stack_id
from optiland.nonsequential.ray_bundle import (
    MEDIUM_STACK_EMPTY,
    MEDIUM_STACK_MAX_DEPTH,
    MediumStackOverflowError,
)
from optiland.nonsequential.rng import EventSlot
from optiland.nonsequential.sampling import resolve_reflect_prob

if TYPE_CHECKING:
    from optiland.coatings import BaseCoating
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.bsdf.base import BaseBSDF
    from optiland.nonsequential.components.geometry.base import ComponentGeometry
    from optiland.nonsequential.ir.bsdf_ir import BsdfIR
    from optiland.nonsequential.ir.scene_ir import SamplingPolicy
    from optiland.nonsequential.materials.nsq_material import NSQMaterial
    from optiland.nonsequential.ray_bundle import NSQRayBundle
    from optiland.nonsequential.rng import NSQRng


class RefractiveComponent(BaseComponent):
    """Refractive optical element (lens, prism, window).

    At each interface, Fresnel splitting uses the detached-sample /
    attached-weight scheme: the branch decision (reflect vs transmit) is
    drawn from a detached probability, while the throughput weight carries
    the attached reflectance so gradients flow through material parameters.

    The two materials name the media on either side of the surface, and the
    component works out which one a ray is leaving by comparing the ray
    direction against the surface's geometric normal (``n_geom``, fixed per
    surface point, pointing from ``material_front`` toward ``material_back``)
    -- never by comparing refractive index values. Crossing direction
    therefore does not matter: the same surface refracts correctly for a ray
    on its way in, for a ghost or retro-reflection coming back through, and
    for the far side of a closed solid modelled as a single geometry, even
    when the two adjacent media have nearly identical indices (a cemented
    doublet, oil immersion).

    Attributes:
        cs: Coordinate system.
        geometry: Surface geometry.
        material_front: Medium on the front (normal-facing) side.
        material_back: Medium on the back side.
        bsdf: Optional BSDF for scatter. None = specular.
        name: Optional label.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        geometry: ComponentGeometry,
        material_front: NSQMaterial,
        material_back: NSQMaterial,
        bsdf: BaseBSDF | None = None,
        name: str = "",
        scatter_fraction: float = 1.0,
        coating: BaseCoating | None = None,
    ) -> None:
        """Initialize RefractiveComponent.

        Args:
            cs: Coordinate system.
            geometry: Surface geometry.
            material_front: Front-side medium. By contract (see
                ``ComponentGeometry.ray_intersect``), this is the medium on
                the side the geometry's *unflipped* normal points away
                from -- for the analytic geometries, the local -z side.
            material_back: Back-side medium -- the side the geometry's
                unflipped normal points toward (local +z, for the analytic
                geometries).
            bsdf: Optional BSDF scatter model.
            name: Optional label.
            scatter_fraction: Probability that a hit ray is routed through
                ``bsdf`` rather than refracted.
            coating: Optional ``optiland.coatings.BaseCoating`` (e.g.
                ``SimpleCoating``). When set, its ``.reflectance``/
                ``.transmittance`` replace the bare Fresnel R/T so NSQ
                agrees with the sequential engine's coating model. Must be
                unpolarized -- a ``BaseCoatingPolarized`` instance raises
                ``NotImplementedError`` immediately, since NSQ rays carry no
                polarization state.
        """
        reject_polarized_coating(coating, surface_name=name)
        self.coating = coating
        super().__init__(
            cs,
            geometry,
            material_front,
            material_back,
            bsdf,
            name,
            scatter_fraction=scatter_fraction,
        )

    def interact(
        self,
        rays: NSQRayBundle,
        t: np.ndarray,
        normals: np.ndarray,
        hit_mask: np.ndarray,
        rng: NSQRng,
        bsdf_ir: BsdfIR,
        n_geom: np.ndarray,
        sampling: SamplingPolicy | None = None,
        forced_branch: str | None = None,
    ) -> None:
        """Apply Fresnel refraction/reflection at hit points (in-place).

        Uses detached-sample / attached-weight Fresnel: the reflect/transmit
        branch decision is drawn from a detached probability so stochastic
        choices do not block gradients; the throughput weight multiplier
        carries the attached reflectance so ∂flux/∂R is non-zero. When
        ``self.coating`` is set, its R/T replace the bare Fresnel values
        (still forced to R=1/T=0 under TIR, where no coating can restore a
        transmitted wave).

        Args:
            rays: Ray bundle updated in-place.
            t: Hit distances [mm], shape (N,).
            normals: Surface normals in global frame, shape (N, 3).
            hit_mask: True for rays hitting this component, shape (N,).
            rng: Keyed PCG32 RNG (used for detached sampling only). Draws
                are keyed by this ray's own id and its bounce count as of
                this interaction, so they are independent of batch_size,
                compaction, and every other ray in the bundle.
            bsdf_ir: This surface's lowered BSDF descriptor. Whether the
                scatter branch below runs at all is decided from
                ``bsdf_ir.kind != "none"`` (verified by the caller to match
                ``self.bsdf``), not from ``self.bsdf is not None``.
            n_geom: Geometric surface normal in global frame, shape (N, 3),
                fixed per surface point and pointing from ``material_front``
                toward ``material_back``. Used, not ``rays.n_current``,
                to determine which material a ray is entering.
            sampling: The scene's rare-path sampling policy.
                Resolves the reflect-branch sampling probability -- see
                :func:`optiland.nonsequential.sampling.resolve_reflect_prob`.
                ``None`` defaults to ``reflect_prob="fresnel"`` (today's
                behaviour). Ignored when ``forced_branch`` is set.
            forced_branch: ``"reflect"`` or ``"transmit"`` to deterministically
                force the branch (weight = R or T exactly, no importance
                division) instead of drawing it stochastically. Used only by
                the NumPy forward engine's bounded-splitting orchestration
                to build both children of a split ray.
        """
        # Every RNG draw in this call is keyed to the ray's identity as of
        # this specific interaction event -- captured before any of the
        # in-place mutations below (including the bounce increment) change
        # rays.bounce out from under us.
        ray_id_key = to_numpy(rays.ray_id)
        bounce_key = to_numpy(rays.bounce)

        # Missed rays carry t = inf; zero it for the differentiable position
        # update so the masked-out be.where branch cannot inject a
        # 0 * inf = NaN into the backward pass.
        t = be.where(hit_mask, t, be.zeros_like(t))

        # Advance hit rays to intersection point
        rays.x = be.where(hit_mask, rays.x + t * rays.L, rays.x)
        rays.y = be.where(hit_mask, rays.y + t * rays.M, rays.y)
        rays.z = be.where(hit_mask, rays.z + t * rays.N, rays.z)

        dirs = be.stack([rays.L, rays.M, rays.N], axis=1)
        wl = rays.wavelength  # µm

        # Determine n1 and n2 for each ray (based on side of the surface)
        dot = (dirs * normals).sum(axis=1)  # signed cos_theta
        cos_theta_i = be.abs(dot)

        # Evaluate the front/back indices at each wavelength -- attached
        # (differentiable).
        n_front = self.material_front.n(wl)
        n_back = self.material_back.n(wl)
        # Extinction coefficients: tracked the same way as n1/n2 below
        # so rays.k_current always reflects the medium a ray is currently
        # travelling through, for Beer-Lambert attenuation on its next hop.
        k_front = self.material_front.k(wl)
        k_back = self.material_back.k(wl)

        # n_geom is fixed per surface point and points from material_front
        # toward material_back (D-1; see ComponentGeometry.ray_intersect),
        # independent of which side the ray approaches from. This replaces
        # the old index-proximity heuristic
        # (`abs(n1 - n2_back) < abs(n1 - n2_front)` against rays.n_current),
        # which silently mis-resolved whenever the two adjacent media had
        # similar indices (a cemented doublet, oil immersion) or the ray
        # took an unexpected path (a ghost re-entering a solid). Comparing
        # ray direction against n_geom is direction-agnostic *and*
        # index-value-agnostic: correct for a ray on its way in, a
        # retro-reflection, or the far side of a closed solid.
        dot_geom = (dirs * n_geom).sum(axis=1)
        entering_back = dot_geom > 0.0
        n1 = be.where(entering_back, n_front, n_back)
        n2 = be.where(entering_back, n_back, n_front)
        k1 = be.where(entering_back, k_front, k_back)
        k2 = be.where(entering_back, k_back, k_front)

        # Fresnel reflectance (unpolarized, attached)
        n_ratio = n1 / (n2 + 1e-30)
        sin2_t = n_ratio**2 * (1.0 - cos_theta_i**2)
        tir = sin2_t > 1.0
        # Epsilon-clamp the radicand (not 0): sqrt's infinite derivative at 0
        # combined with be.where yields a 0 * inf = NaN gradient at the TIR
        # boundary. Forward value changes by <= 1e-6.
        cos_theta_t = be.where(
            tir,
            be.zeros_like(sin2_t),
            be.maximum(1.0 - sin2_t, 1e-12) ** 0.5,
        )

        rs = (n1 * cos_theta_i - n2 * cos_theta_t) / (
            n1 * cos_theta_i + n2 * cos_theta_t + 1e-30
        )
        rp = (n2 * cos_theta_i - n1 * cos_theta_t) / (
            n2 * cos_theta_i + n1 * cos_theta_t + 1e-30
        )
        R_fresnel = be.where(tir, be.ones_like(rs), 0.5 * (rs**2 + rp**2))

        # A coating overrides the bare Fresnel R/T with its own (possibly
        # wavelength-independent, possibly lossy: R + T < 1) values -- except
        # under TIR, where there is no real transmitted wave regardless of
        # what the coating claims, so reflection stays forced to R=1, T=0.
        if self.coating is not None:
            R_used = be.ones_like(R_fresnel) * float(self.coating.reflectance)
            T_used = be.ones_like(R_fresnel) * float(self.coating.transmittance)
        else:
            R_used = R_fresnel
            T_used = 1.0 - R_fresnel
        R_used = be.where(tir, be.ones_like(R_used), R_used)
        T_used = be.where(tir, be.zeros_like(T_used), T_used)

        # --- Detached-sample / attached-weight ---
        R_np = to_numpy(R_used).astype(np.float64)

        if forced_branch is not None:
            # Bounded-splitting orchestration (PR11, NumPy forward engine
            # only): the branch is fixed, not drawn, and the weight is the
            # exact deterministic R or T -- no importance division, since
            # there is no probability being compensated for. TIR rays are
            # unaffected: T_used is already forced to 0 there, so a forced
            # "transmit" branch on a TIR ray correctly carries zero flux
            # rather than raising or fabricating a wave that cannot exist.
            do_reflect_np = np.full_like(R_np, forced_branch == "reflect", dtype=bool)
            do_reflect = be.array(do_reflect_np)
            weight = be.where(do_reflect, R_used, T_used)
        else:
            # Importance-biased branch probability: generalises
            # the plain-Fresnel estimator (p == R) to any detached
            # probability p, dividing by p rather than R_det so the
            # estimator stays unbiased for any p in (0, 1) -- only the
            # variance changes. reflect_prob="fresnel" reproduces the
            # original weight formula exactly.
            r_det = be.array(R_np)
            p_be = resolve_reflect_prob(sampling, r_det) if sampling else r_det
            p_np = np.clip(to_numpy(p_be).astype(np.float64), 1e-12, 1.0 - 1e-12)
            u = rng.uniform(ray_id_key, bounce_key, EventSlot.FRESNEL_BRANCH)
            do_reflect_np = (u < p_np) | to_numpy(tir).astype(bool)
            do_reflect = be.array(do_reflect_np)

            # Throughput weight: forward value is 1.0 in expectation; carries
            # gradients through R/T. Generalizes the plain-Fresnel
            # weight_transmit = (1-R)/(1-R_det) to allow T != 1-R (coating
            # absorption) *and* p != R (importance biasing): the branch is
            # still a single reflect-vs-transmit draw with P(reflect) = p,
            # so E[weight] = R on the reflect branch and T on the transmit
            # branch regardless of p -- exact flux conservation in
            # expectation, with the shortfall R+T<1 taken up by the
            # deterministic T weight rather than a separate absorption draw.
            # For TIR rays weight stays 1.0 (full reflection is deterministic).
            p_det = be.array(p_np)  # detached copy used as denominator
            weight_reflect = R_used / (p_det + 1e-30)
            weight_transmit = T_used / (1.0 - p_det + 1e-30)
            weight = be.where(do_reflect, weight_reflect, weight_transmit)
            # TIR: weight is exactly 1
            weight = be.where(tir, be.ones_like(weight), weight)

        # Apply weight to flux for hit rays
        rays.flux = rays.flux * be.where(hit_mask, weight, be.ones_like(weight))

        # Compute reflected direction: d - 2*(d.n)*n
        raw_dot = (dirs * normals).sum(axis=1, keepdims=True)
        reflected = dirs - 2.0 * raw_dot * normals
        norms_r = (reflected * reflected).sum(axis=1, keepdims=True) ** 0.5
        reflected = reflected / (norms_r + 1e-30)

        # Compute refracted direction (Snell's law, vector form)
        n_facing = be.where(raw_dot < 0, normals, -normals)
        cos_i_pos = be.abs(raw_dot)
        n_ratio_col = n_ratio[:, None]
        cos_t_col = cos_theta_t[:, None]
        refracted = (
            n_ratio_col * dirs + (n_ratio_col * cos_i_pos - cos_t_col) * n_facing
        )
        norms_t = (refracted * refracted).sum(axis=1, keepdims=True) ** 0.5
        refracted = refracted / (norms_t + 1e-30)

        # Select direction based on branch decision
        do_reflect_col = do_reflect[:, None]
        new_d = be.where(do_reflect_col, reflected, refracted)

        # Apply only to hit rays
        hit_col = hit_mask[:, None]
        rays.L = be.where(hit_col[:, 0], new_d[:, 0], rays.L)
        rays.M = be.where(hit_col[:, 0], new_d[:, 1], rays.M)
        rays.N = be.where(hit_col[:, 0], new_d[:, 2], rays.N)

        # Update n_current/k_current: stays medium 1 on reflect, becomes
        # medium 2 on refract.
        rays.n_current = be.where(
            hit_mask, be.where(do_reflect, n1, n2), rays.n_current
        )
        rays.k_current = be.where(
            hit_mask, be.where(do_reflect, k1, k2), rays.k_current
        )

        # D1: medium stack push/pop -- a diagnostic cross-check, never fed
        # back into n1/n2 above. Direction is decided by medium identity,
        # not entering_back (a Lens's two faces share one +n_geom
        # convention but opposite interior sides). Reaching ambient (id 0)
        # always unwinds the whole stack, since abutting media (e.g. a
        # cemented doublet) push sequentially without true nesting; a pop
        # at depth 0 is counted as a leak. Reaching a non-ambient medium
        # that matches one level below the top is a true nesting exit
        # (pop); anything else pushes.
        transmit_np = to_numpy(hit_mask).astype(bool) & ~to_numpy(do_reflect).astype(
            bool
        )
        if transmit_np.any():
            entering_back_np = to_numpy(entering_back).astype(bool)
            front_id = medium_stack_id(self.material_front)
            back_id = medium_stack_id(self.material_back)
            mat2_np = np.where(entering_back_np, back_id, front_id)

            rows = np.where(transmit_np)[0]
            mat2_rows = mat2_np[rows]
            ambient_mask = mat2_rows == 0

            if ambient_mask.any():
                amb_rows = rows[ambient_mask]
                underflow_mask = rays.medium_depth[amb_rows] == 0
                if underflow_mask.any():
                    rays.medium_stack_underflows[amb_rows[underflow_mask]] += 1
                rays.medium_stack[amb_rows, :] = MEDIUM_STACK_EMPTY
                rays.medium_depth[amb_rows] = 0

            non_ambient_mask = ~ambient_mask
            if non_ambient_mask.any():
                na_rows = rows[non_ambient_mask]
                na_mat2 = mat2_rows[non_ambient_mask]
                na_depth = rays.medium_depth[na_rows]

                below = np.zeros_like(na_mat2)  # depth 0 or 1 -> "below" is ambient
                two_or_more = na_depth >= 2
                if two_or_more.any():
                    below[two_or_more] = rays.medium_stack[
                        na_rows[two_or_more], na_depth[two_or_more] - 2
                    ]
                pop_mask = (na_depth >= 1) & (na_mat2 == below)
                push_mask = ~pop_mask

                if pop_mask.any():
                    pr = na_rows[pop_mask]
                    rays.medium_depth[pr] -= 1
                    rays.medium_stack[pr, rays.medium_depth[pr]] = MEDIUM_STACK_EMPTY
                if push_mask.any():
                    pr = na_rows[push_mask]
                    if (rays.medium_depth[pr] >= MEDIUM_STACK_MAX_DEPTH).any():
                        raise MediumStackOverflowError(
                            f"Medium stack exceeded MEDIUM_STACK_MAX_DEPTH="
                            f"{MEDIUM_STACK_MAX_DEPTH} at surface "
                            f"{self.name or type(self).__name__!r}. This "
                            "indicates either pathologically deep volume "
                            "nesting or a geometry defect that pushes "
                            "without popping."
                        )
                    rays.medium_stack[pr, rays.medium_depth[pr]] = na_mat2[push_mask]
                    rays.medium_depth[pr] += 1

        # Update bounce count
        rays.bounce = be.where(hit_mask, rays.bounce + 1, rays.bounce)

        # Apply BSDF scatter if present (compute for all rays, use where to select)
        if bsdf_ir.kind != "none":
            # Compute BSDF for all N rays; where-select only hit rays
            bsdf_dirs, bsdf_weights, bsdf_transmitted = self.bsdf.sample(
                rays.num_rays,
                be.stack([rays.L, rays.M, rays.N], axis=1),
                normals,
                rays.wavelength,
                rng,
                ray_id_key,
                bounce_key,
            )
            # Route only a scatter_fraction of the hit rays through the BSDF;
            # the rest keep the refracted direction computed above. The
            # branch is drawn from a detached probability, matching the
            # Fresnel split -- and, like the Fresnel split, carries a
            # compensating attached weight so d(flux)/d(scatter_fraction) is
            # correct rather than silently zero. Epsilon-clamped denominator
            # for the same reason as the Fresnel branch: scatter_fraction=1
            # (or 0) exactly would otherwise divide by zero for the ~1e-6
            # fraction of draws the clamp itself puts on the "wrong" side.
            sf_det = float(np.clip(to_numpy(self.scatter_fraction), 1e-6, 1.0 - 1e-6))
            u_scatter = rng.uniform(ray_id_key, bounce_key, EventSlot.SCATTER_BRANCH)
            scatters_np = to_numpy(hit_mask).astype(bool) & (u_scatter < sf_det)
            scatters = be.array(scatters_np)

            sf = self.scatter_fraction
            weight_scatter_branch = sf / sf_det
            weight_nonscatter_branch = (1.0 - sf) / (1.0 - sf_det)
            sf_gate = be.where(
                scatters, weight_scatter_branch, weight_nonscatter_branch
            )
            rays.flux = rays.flux * be.where(hit_mask, sf_gate, be.ones_like(sf_gate))

            scatter_col = scatters[:, None]
            cur_dirs = be.stack([rays.L, rays.M, rays.N], axis=1)
            new_dirs = be.where(scatter_col, bsdf_dirs, cur_dirs)
            rays.L = new_dirs[:, 0]
            rays.M = new_dirs[:, 1]
            rays.N = new_dirs[:, 2]
            bsdf_gate = be.where(scatters, bsdf_weights, be.ones_like(bsdf_weights))
            rays.flux = rays.flux * bsdf_gate

            # D-4: a scattered ray's medium is decided by its own lobe's
            # reflect/transmit side, not by the Fresnel branch draw above --
            # that draw only describes what happens to a ray that does NOT
            # enter the BSDF lobe. Re-resolves n_current/k_current for
            # exactly the scattered rays.
            bsdf_in_medium2 = be.array(bsdf_transmitted)
            rays.n_current = be.where(
                scatters, be.where(bsdf_in_medium2, n2, n1), rays.n_current
            )
            rays.k_current = be.where(
                scatters, be.where(bsdf_in_medium2, k2, k1), rays.k_current
            )
            # Note: the medium stack above is not re-synced for scattered
            # rays -- a transmissive BSDF lobe can cross the boundary
            # opposite to the deterministic Fresnel branch's push/pop, so
            # scatter_fraction > 0 on a volume boundary can drift the stack
            # out of sync with n_current (diagnostic-only; n1/n2 stay
            # correct either way).
