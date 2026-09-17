"""Owned calculation inputs and revision-tagged results (no Qt objects)."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from types import MethodType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optiland.optic import Optic


@dataclass(frozen=True)
class DocumentToken:
    """Identity and optical revision of a GUI-owned document."""

    document_id: str
    revision: int


@dataclass(frozen=True)
class BackendConfig:
    """Backend configuration explicitly installed in the calculation process."""

    name: str = "numpy"
    device: str = "cpu"
    precision: int = 64

    @classmethod
    def capture(cls) -> BackendConfig:
        import optiland.backend as be

        name = be.get_backend()
        if name == "torch":
            return cls(name, be.get_device(), be.get_precision())
        return cls(name)

    def apply(self) -> None:
        import optiland.backend as be

        be.set_backend(self.name)
        if self.name == "torch":
            be.set_device(self.device)
            be.set_precision(f"float{self.precision}")


@dataclass(frozen=True)
class OpticSnapshot:
    """Detached prescription bytes; capturing does not run the optic updater.

    The internal pickle transports our own serializer output, including existing
    polarization records and arrays. It is never loaded from an external file.
    The public Optiland data format remains unchanged.
    """

    data: bytes
    backend: BackendConfig
    component_types: tuple = ()

    @classmethod
    def capture(cls, optic: Optic) -> OpticSnapshot:
        from optiland.optic import Optic

        if any(
            getattr(type(optic), name) is not getattr(Optic, name)
            for name in ("trace", "trace_generic")
        ):
            raise ValueError(
                "This custom tracing model needs an owned snapshot adapter."
            )
        for component, methods in (
            (optic, ("trace", "trace_generic")),
            (optic.ray_tracer, ("trace", "trace_generic")),
            (
                getattr(optic.ray_tracer, "ray_generator", None),
                ("generate_rays",),
            ),
        ):
            if component is None:
                continue
            for name in methods:
                if name not in vars(component):
                    continue
                method = vars(component)[name]
                # Restoring an instrumented method may bind the unchanged class
                # implementation on the instance. Only that exact same-owner
                # method is equivalent to what the serializer reconstructs.
                if not (
                    isinstance(method, MethodType)
                    and method.__self__ is component
                    and method.__func__ is getattr(type(component), name, None)
                ):
                    raise ValueError(
                        "This custom tracing model needs an owned snapshot adapter."
                    )
        component_types = _component_types(optic)
        return cls(
            pickle.dumps(optic.to_dict(), protocol=5),
            BackendConfig.capture(),
            component_types,
        )

    def restore(self) -> Optic:
        from optiland.optic import Optic

        self.backend.apply()
        optic = Optic.from_dict(pickle.loads(self.data))
        if self.component_types and _component_types(optic) != self.component_types:
            raise ValueError(
                "Snapshot reconstruction changed an optical component type."
            )
        return optic


def _component_types(optic: Optic) -> tuple:
    """Reject unsupported runtime extensions rather than silently losing their type."""
    components = [
        optic.aperture,
        optic.apodization,
        optic.fields.field_definition,
        optic.ray_tracer,
        getattr(optic.ray_tracer, "ray_generator", None),
    ]
    for surface in optic.surfaces:
        components.extend(
            (
                surface,
                surface.geometry,
                surface.geometry.cs,
                surface.material_post,
                surface.aperture,
                surface.interaction_model,
                getattr(surface.interaction_model, "coating", None),
                getattr(surface, "source", None),
            )
        )
    components.extend(optic.pickups.pickups)
    components.extend(optic.solves.solves)
    records = []
    for component in components:
        if component is None:
            records.append(None)
            continue
        cls = type(component)
        if not cls.__module__.startswith("optiland."):
            raise ValueError(
                f"Custom component {cls.__qualname__} needs a snapshot adapter."
            )
        records.append((cls.__module__, cls.__qualname__))
    return tuple(records)


@dataclass(frozen=True)
class JobRequest:
    """One calculation target generation and its detached inputs.

    ``context`` belongs to the GUI only (for example captured surface identities).
    The process transport deliberately excludes it.
    """

    job_id: int
    document: DocumentToken
    target: str
    generation: int
    handler: str
    snapshot: OpticSnapshot | None
    parameters: dict[str, Any]
    cancel_on_document_change: bool = True
    context: Any = field(default=None, compare=False, repr=False)

    def worker_message(self) -> dict:
        return {
            "command": "run",
            "job_id": self.job_id,
            "handler": self.handler,
            "snapshot": self.snapshot,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class JobResult:
    """Terminal calculation outcome, with GUI-side freshness determined once."""

    request: JobRequest
    status: str
    data: Any = None
    error: str = ""
    current: bool = False
    infrastructure_error: bool = False


class CalculationCancelled(Exception):
    """Raised at a safe numerical checkpoint after cancellation."""


def check_cancelled(cancelled: Any) -> None:
    """Stop numerical work at a safe boundary if cancellation was requested."""
    if cancelled.is_set():
        raise CalculationCancelled()
