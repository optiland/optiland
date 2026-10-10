"""Private, owned transport for an already reconstructed built-in optical model.

These bytes are generated only by our calculation worker, never read from a
user file. Public saved prescriptions continue to use ``Optic.to_dict``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

from optiland.optic import Optic
from optiland_gui.services.job_records import BackendConfig, _component_types


@dataclass(frozen=True)
class PreparedOptic:
    """An internal built-in model plus its explicit backend and type manifest."""

    data: bytes
    backend: BackendConfig
    component_types: tuple

    @classmethod
    def capture(cls, optic):
        if type(optic) is not Optic or "trace" in vars(optic):
            raise ValueError("A custom model needs an owned preparation adapter.")
        return cls(
            pickle.dumps(optic, protocol=5),
            BackendConfig.capture(),
            _component_types(optic),
        )

    def restore(self):
        if BackendConfig.capture() != self.backend:
            raise ValueError("The calculation backend changed during preparation.")
        optic = pickle.loads(self.data)
        if type(optic) is not Optic or _component_types(optic) != self.component_types:
            raise ValueError("Prepared optical model has unexpected component types.")
        return optic
