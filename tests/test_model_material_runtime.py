from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import Material, ModelMaterial

from .utils import assert_allclose


LAMBDA_D = 0.5875618
LAMBDA_F = 0.4861327
LAMBDA_C = 0.6562725
LAMBDA_G = 0.4358343
LAMBDA_REF = 1.55

RUNTIME_WAVELENGTHS = np.array(
    [
        0.36501,
        0.40466,
        LAMBDA_G,
        LAMBDA_F,
        LAMBDA_D,
        LAMBDA_C,
        1.014,
        LAMBDA_REF,
        2.3,
    ],
)

CATALOG_CASES = [
    ("schott", "N-BK7"),
    ("ohara", "S-FPL53"),
]


def _scalar(value) -> float:
    """Return a backend-agnostic scalar as a Python float."""
    return float(np.asarray(be.to_numpy(value)).reshape(-1)[0])


def _catalog_descriptors(catalog: Material) -> tuple[float, float, float]:
    """Compute nd, vd, and dPgF from catalog Fraunhofer indices."""
    nd = _scalar(catalog.n(LAMBDA_D))
    n_f = _scalar(catalog.n(LAMBDA_F))
    n_c = _scalar(catalog.n(LAMBDA_C))
    n_g = _scalar(catalog.n(LAMBDA_G))

    vd = (nd - 1.0) / (n_f - n_c)
    pgf = (n_g - n_f) / (n_f - n_c)
    dpgf = pgf - (0.6438 - 0.001682 * vd)

    return nd, vd, dpgf


def _catalog_p_ref(catalog: Material, nd: float, vd: float) -> float:
    """Compute the reference partial dispersion from the catalog index."""
    n_ref = _scalar(catalog.n(LAMBDA_REF))

    return (n_ref - nd) / ((nd - 1.0) / vd)


def _catalog_indices(catalog: Material, wavelengths: np.ndarray) -> np.ndarray:
    """Return catalog refractive indices as NumPy values."""
    return np.asarray(be.to_numpy(catalog.n(be.array(wavelengths))), dtype=float)


@pytest.mark.parametrize(("reference", "name"), CATALOG_CASES)
def test_visible_route_matches_catalog_runtime_validation_glasses(
    reference,
    name,
    set_test_backend,
):
    catalog = Material(name, reference=reference)
    nd, vd, dpgf = _catalog_descriptors(catalog)
    model = ModelMaterial(nd, vd, dpgf)

    truth = _catalog_indices(catalog, RUNTIME_WAVELENGTHS)
    predicted = np.asarray(
        be.to_numpy(model.n(be.array(RUNTIME_WAVELENGTHS))),
        dtype=float,
    )

    assert np.max(np.abs(predicted - truth)) < 1e-3


@pytest.mark.parametrize(("reference", "name"), CATALOG_CASES)
def test_ir_route_matches_catalog_runtime_validation_glasses(
    reference,
    name,
    set_test_backend,
):
    catalog = Material(name, reference=reference)
    nd, vd, dpgf = _catalog_descriptors(catalog)
    p_ref = _catalog_p_ref(catalog, nd, vd)
    model = ModelMaterial(
        nd,
        vd,
        dpgf,
        P_ref=p_ref,
        lambda_ref=LAMBDA_REF,
    )

    truth = _catalog_indices(catalog, RUNTIME_WAVELENGTHS)
    predicted = np.asarray(
        be.to_numpy(model.n(be.array(RUNTIME_WAVELENGTHS))),
        dtype=float,
    )

    assert_allclose(model.n(LAMBDA_REF), catalog.n(LAMBDA_REF), atol=1e-9)
    assert np.max(np.abs(predicted - truth)) < 2e-4
