"""Vectorised columnar path recording.

The pre-PR12 implementation built one Python ``dict`` per event in a
``for k in range(n)`` loop, appended it to a list, then copied it
field-by-field into a structured array a second time -- plus nine separate
``to_numpy()`` round trips per log call. At 10^6 rays x 4 events this took
minutes and gigabytes of RAM, on exactly the code path a user reaches for
when they need to know where stray light came from.

This module replaces that with :class:`ColumnarPathLog`: preallocated,
amortised-growth NumPy arrays, appended via a single fancy-index write per
log call (no per-event Python objects), with the final structured array
built once, at the end of the trace, by vectorised dtype/lookup conversion
-- not incrementally.

:class:`PathRecorder` additionally implements the ``record_paths: int``
contract : recording a uniformly-chosen random subset of that many
rays rather than every ray, selected by a PCG32 hash of ``ray_id`` (see
:mod:`optiland.nonsequential.rng`) so the subset is deterministic under the
RNG contract -- independent of ``batch_size`` and of NumPy-vs-Torch
compaction -- and stable for a given ``seed`` regardless of how many rays
end up actually being born (a source can generate fewer rays than asked).

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optiland.backend.utils import to_numpy
from optiland.nonsequential.rng import EventSlot, pcg32_uint32

if TYPE_CHECKING:
    from optiland.nonsequential.ray_bundle import NSQRayBundle

# Structured dtype the rest of the package (visualization, tests) consumes.
# Kept stable across the D-7 rewrite: only the *internal* accumulation
# strategy changed, not the format handed to consumers.
_EVENT_DTYPE = np.dtype(
    [
        ("ray_id", np.int64),
        ("event_type", "U10"),
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("L", np.float64),
        ("M", np.float64),
        ("N", np.float64),
        ("flux", np.float64),
        ("wavelength", np.float64),
        ("bounce", np.int32),
        ("component_name", "U64"),
    ]
)

_EVENT_TYPE_NAMES = np.array(["birth", "hit", "death"])
_BIRTH, _HIT, _DEATH = 0, 1, 2

# Columnar fields as (attribute, dtype) pairs, shared by allocation, growth,
# and the final structured-array assembly.
_FIELDS: tuple[tuple[str, type], ...] = (
    ("_ray_id", np.int64),
    ("_event_type", np.int8),
    ("_x", np.float64),
    ("_y", np.float64),
    ("_z", np.float64),
    ("_L", np.float64),
    ("_M", np.float64),
    ("_N", np.float64),
    ("_flux", np.float64),
    ("_wavelength", np.float64),
    ("_bounce", np.int32),
    ("_name_id", np.int32),
)


def resolve_path_sample_mask(
    ray_id: np.ndarray,
    num_rays_total: int,
    record_paths: bool | int,
    seed: int,
) -> np.ndarray:
    """Which of ``ray_id`` fall in the recorded subset.

    Args:
        ray_id: Per-ray identifiers, shape (N,).
        num_rays_total: Total rays the trace will launch (``ray_id`` values
            range over ``[0, num_rays_total)``).
        record_paths: ``False``/``0`` records nothing; ``True`` records
            every ray; a positive ``int`` records an approximately
            ``record_paths``-sized subset, selected by a PCG32 hash of
            ``ray_id`` so the *same* rays are selected regardless of
            ``batch_size`` or backend.
        seed: Trace-level RNG seed (the hash is keyed by it, so the sampled
            subset changes with the seed like every other stochastic
            decision in the engine).

    Returns:
        Boolean NumPy array, shape matching ``ray_id``.
    """
    ray_id_np = np.asarray(to_numpy(ray_id))
    if record_paths is True:
        return np.ones(ray_id_np.shape, dtype=bool)
    if not record_paths or num_rays_total <= 0:
        return np.zeros(ray_id_np.shape, dtype=bool)
    n = int(record_paths)
    if n >= num_rays_total:
        return np.ones(ray_id_np.shape, dtype=bool)

    # A ray's hash is a pure function of (seed, ray_id) -- independent of
    # bounce or batch -- so the same ray is always in or always out of the
    # sample for its whole lifetime, across every event type.
    h = pcg32_uint32(seed, ray_id_np, np.zeros_like(ray_id_np), EventSlot.PATH_SAMPLE)
    # threshold = (n / num_rays_total) * 2**32, computed to stay in uint32
    # range without overflowing an intermediate float32.
    threshold = np.uint32(min(int((n / num_rays_total) * 2.0**32), 0xFFFFFFFF))
    return h < threshold


class ColumnarPathLog:
    """Preallocated, amortised-growth columnar event log.

    Every ``log_event()`` call is one vectorised fancy-index write into
    preallocated NumPy arrays -- no Python loop over rays, no per-event
    dict. Capacity doubles when exceeded (standard amortised-growth
    strategy), so total copying cost across a trace is O(final size), not
    O(final size^2). ``event_type`` and ``component_name`` are stored as
    small integer codes during accumulation and expanded to strings only
    once, in :meth:`to_events`, via vectorised fancy indexing -- not per
    row.
    """

    def __init__(self, initial_capacity: int = 4096) -> None:
        """Initialize an empty log.

        Args:
            initial_capacity: Starting buffer size; grows by doubling.
        """
        self._capacity = max(int(initial_capacity), 1)
        self._count = 0
        for attr, dtype in _FIELDS:
            setattr(self, attr, np.empty(self._capacity, dtype=dtype))
        self._names: list[str] = []
        self._name_index: dict[str, int] = {}

    def _name_id_for(self, name: str) -> int:
        """Return (assigning if new) the integer code for ``name``."""
        idx = self._name_index.get(name)
        if idx is None:
            idx = len(self._names)
            self._names.append(name)
            self._name_index[name] = idx
        return idx

    def _ensure_capacity(self, extra: int) -> None:
        needed = self._count + extra
        if needed <= self._capacity:
            return
        new_capacity = max(self._capacity * 2, needed)
        for attr, dtype in _FIELDS:
            old = getattr(self, attr)
            new = np.empty(new_capacity, dtype=dtype)
            new[: self._count] = old[: self._count]
            setattr(self, attr, new)
        self._capacity = new_capacity

    def log_event(
        self,
        event_type_code: int,
        mask: np.ndarray,
        rays: NSQRayBundle,
        t_offset: object | None,
        name: str,
    ) -> None:
        """Append one event per True row of ``mask``, vectorised.

        Args:
            event_type_code: One of :data:`_BIRTH`, :data:`_HIT`,
                :data:`_DEATH`.
            mask: Boolean array/tensor, shape (N,); rows to log.
            rays: Ray bundle to read positions/directions/flux from.
            t_offset: If given, advance the logged position by
                ``t_offset * direction`` (the hit point; positions have not
                yet been advanced to it when a component logs its hit).
                ``None`` logs the position as-is (birth, death -- already
                advanced by the caller).
            name: Component/source/death-cause label for every logged row.
        """
        mask_np = np.asarray(to_numpy(mask), dtype=bool)
        idx = np.where(mask_np)[0]
        k = idx.size
        if k == 0:
            return

        x_np = to_numpy(rays.x)[idx]
        y_np = to_numpy(rays.y)[idx]
        z_np = to_numpy(rays.z)[idx]
        L_np = to_numpy(rays.L)[idx]
        M_np = to_numpy(rays.M)[idx]
        N_np = to_numpy(rays.N)[idx]
        if t_offset is not None:
            t_np = to_numpy(t_offset)[idx]
            x_np = x_np + t_np * L_np
            y_np = y_np + t_np * M_np
            z_np = z_np + t_np * N_np
        flux_np = to_numpy(rays.flux)[idx]
        wl_np = to_numpy(rays.wavelength)[idx]
        bounce_np = to_numpy(rays.bounce)[idx]
        ray_id_np = to_numpy(rays.ray_id)[idx]

        self._ensure_capacity(k)
        s = slice(self._count, self._count + k)
        self._ray_id[s] = ray_id_np
        self._event_type[s] = event_type_code
        self._x[s] = x_np
        self._y[s] = y_np
        self._z[s] = z_np
        self._L[s] = L_np
        self._M[s] = M_np
        self._N[s] = N_np
        self._flux[s] = flux_np
        self._wavelength[s] = wl_np
        self._bounce[s] = bounce_np
        self._name_id[s] = self._name_id_for(name)
        self._count += k

    def to_events(self) -> np.ndarray | None:
        """Build the final structured event array, once.

        Returns:
            A :data:`_EVENT_DTYPE` structured array, or ``None`` if nothing
            was ever logged.
        """
        if self._count == 0:
            return None
        n = self._count
        arr = np.zeros(n, dtype=_EVENT_DTYPE)
        arr["ray_id"] = self._ray_id[:n]
        arr["event_type"] = _EVENT_TYPE_NAMES[self._event_type[:n]]
        arr["x"] = self._x[:n]
        arr["y"] = self._y[:n]
        arr["z"] = self._z[:n]
        arr["L"] = self._L[:n]
        arr["M"] = self._M[:n]
        arr["N"] = self._N[:n]
        arr["flux"] = self._flux[:n]
        arr["wavelength"] = self._wavelength[:n]
        arr["bounce"] = self._bounce[:n]
        names_arr = np.array(self._names) if self._names else np.array([], dtype="<U1")
        arr["component_name"] = names_arr[self._name_id[:n]]
        return arr


class PathRecorder:
    """Trace-scoped facade combining subset sampling with the columnar log.

    Both reference backends construct one of these per ``trace()`` call and
    use its ``log_birth``/``log_hits``/``log_deaths``/``finalize`` methods
    in place of the old per-backend ``_log_birth``/``_log_hits``/
    ``_log_deaths`` closures -- identical usage on both backends, so this
    is also where that duplication was consolidated.
    """

    def __init__(
        self, record_paths: bool | int, num_rays_total: int, seed: int
    ) -> None:
        """Initialize a PathRecorder.

        Args:
            record_paths: See :func:`resolve_path_sample_mask`.
            num_rays_total: Total rays the trace will launch.
            seed: Trace-level RNG seed.
        """
        self.enabled = bool(record_paths)
        self._record_paths = record_paths
        self._num_rays_total = num_rays_total
        self._seed = seed
        self._log = ColumnarPathLog() if self.enabled else None

    def _sample_mask(self, rays: NSQRayBundle) -> np.ndarray:
        return resolve_path_sample_mask(
            rays.ray_id, self._num_rays_total, self._record_paths, self._seed
        )

    def log_birth(self, rays: NSQRayBundle, source_name: str) -> None:
        """Log a birth event for every ray in ``rays`` that is in-sample."""
        if not self.enabled:
            return
        self._log.log_event(_BIRTH, self._sample_mask(rays), rays, None, source_name)

    def log_hits(
        self,
        rays: NSQRayBundle,
        mask: np.ndarray,
        name: str,
        t_offset: object | None = None,
    ) -> None:
        """Log a hit event for in-sample rays where ``mask`` is True.

        Matches the ``LogHitFn`` contract in
        :mod:`optiland.nonsequential.ir.interpreter`, so a bound method of
        this class can be passed directly as ``log_hit_fn`` /
        ``apply_primitive_interactions``'s detector-hit logging.
        """
        if not self.enabled:
            return
        mask_np = np.asarray(to_numpy(mask), dtype=bool)
        combined = mask_np & self._sample_mask(rays)
        self._log.log_event(_HIT, combined, rays, t_offset, name)

    def log_deaths(self, rays: NSQRayBundle, mask: np.ndarray, cause: str) -> None:
        """Log a death event for in-sample rays where ``mask`` is True."""
        if not self.enabled:
            return
        mask_np = np.asarray(to_numpy(mask), dtype=bool)
        combined = mask_np & self._sample_mask(rays)
        self._log.log_event(_DEATH, combined, rays, None, cause)

    def finalize(self) -> dict | None:
        """Return ``{"events": structured_array}``, or ``None`` if empty.

        Returns:
            The recorded log in the format
            :class:`~optiland.nonsequential.tracer.SimulationResult.ray_paths`
            has always used, or ``None`` if recording was disabled or
            nothing was logged.
        """
        if not self.enabled:
            return None
        events = self._log.to_events()
        if events is None:
            return None
        return {"events": events}
