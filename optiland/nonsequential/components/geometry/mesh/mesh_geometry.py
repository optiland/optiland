"""Mesh geometry for Non-Sequential Raytracing.

Triangulated surface backed by trimesh. Uses trimesh's built-in BVH for
CPU-side ray intersection.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

from optiland.nonsequential.components.geometry.base import AABB, AnalyticGeometry


class MeshGeometry(AnalyticGeometry):
    """Triangulated surface backed by a trimesh.Trimesh object.

    Intersection uses trimesh's ray caster (pyembree if installed, else
    the pure-Python fallback). GPU transfer is required on each intersection
    step for CuPy arrays; analytic geometry is preferred for GPU performance.

    Attributes:
        mesh: The underlying trimesh.Trimesh object.
    """

    def __init__(self, mesh: object) -> None:  # trimesh.Trimesh
        """Initialize MeshGeometry.

        Args:
            mesh: A trimesh.Trimesh instance defining the surface.

        Raises:
            ImportError: If trimesh is not installed.
        """
        try:
            import trimesh  # noqa: F401  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "trimesh is required for MeshGeometry. "
                "Install with: pip install trimesh"
            ) from e
        self.mesh = mesh

    def ray_intersect(
        self, origins: np.ndarray, directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the mesh using trimesh BVH.

        Arrays are converted to NumPy for trimesh, then results are
        converted back to the original array type (CuPy if needed).

        Args:
            origins: Ray origins in local frame, shape (N, 3) [mm].
            directions: Ray directions in local frame, shape (N, 3).

        Returns:
            (t, normals, hit_mask, n_geom). n_geom is trimesh's raw
            ``face_normals`` value, i.e. the ``material_back`` side by
            contract (see :meth:`ComponentGeometry.ray_intersect`) is
            whichever side the mesh's face winding points away from --
            consistently outward for a properly wound (CCW, right-hand
            rule) closed mesh.
        """
        xp = _get_xp(origins)
        # Bring to CPU for trimesh
        o_np = _to_numpy(origins)
        d_np = _to_numpy(directions)
        N = o_np.shape[0]

        # trimesh ray casting
        locations, ray_indices, triangle_indices = self.mesh.ray.intersects_location(
            ray_origins=o_np,
            ray_directions=d_np,
            multiple_hits=False,
        )

        t_out = np.full(N, np.inf, dtype=np.float64)
        normals_out = np.zeros((N, 3), dtype=np.float64)
        n_geom_out = np.zeros((N, 3), dtype=np.float64)

        if len(ray_indices) > 0:
            # Compute t for each hit
            hit_vecs = locations - o_np[ray_indices]
            # t = dot(hit_vec, direction) / |direction|^2 ~ dot for unit vectors
            t_vals = (hit_vecs * d_np[ray_indices]).sum(axis=1)
            # Keep nearest positive hit per ray
            order = np.argsort(ray_indices)
            for idx, ri in enumerate(ray_indices[order]):
                tv = t_vals[order[idx]]
                if tv > 1e-9 and tv < t_out[ri]:
                    t_out[ri] = tv
                    tri_idx = triangle_indices[order[idx]]
                    face_normal = self.mesh.face_normals[tri_idx]
                    n_geom_out[ri] = face_normal
                    # Flip to face incoming ray
                    if np.dot(d_np[ri], face_normal) > 0:
                        face_normal = -face_normal
                    normals_out[ri] = face_normal

        hit_mask_np = t_out < np.inf

        if xp is not np:
            t_out = xp.array(t_out)
            normals_out = xp.array(normals_out)
            hit_mask_np = xp.array(hit_mask_np)
            n_geom_out = xp.array(n_geom_out)

        return t_out, normals_out, hit_mask_np, n_geom_out

    def bounding_box(self, transform: tuple[np.ndarray, np.ndarray]) -> AABB:
        """Return AABB for the mesh in global coordinates.

        Args:
            transform: (translation, rotation_matrix).

        Returns:
            AABB in global frame.
        """
        t_vec = np.array(transform[0], dtype=float)
        R = np.array(transform[1], dtype=float)
        verts_local = np.array(self.mesh.vertices, dtype=float)
        verts_global = verts_local @ R.T + t_vec
        return AABB(verts_global.min(axis=0), verts_global.max(axis=0))


def _to_numpy(arr: np.ndarray) -> np.ndarray:
    try:
        import cupy  # type: ignore[import]

        if isinstance(arr, cupy.ndarray):
            return cupy.asnumpy(arr)
    except ImportError:
        pass
    return arr


def _get_xp(arr: np.ndarray):
    try:
        import cupy  # type: ignore[import]

        if isinstance(arr, cupy.ndarray):
            return cupy
    except ImportError:
        pass
    return np
