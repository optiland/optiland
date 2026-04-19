"""Thin film optics core functions.

This provides core functions for thin film optics calculations using the
transfer matrix method (TMM).

Corentin Nannini, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import optiland.backend as be

if TYPE_CHECKING:
    from optiland.materials import BaseMaterial
    from optiland.thin_film import ThinFilmStack

Array: TypeAlias = Any  # be.ndarray
PolSP = Literal["s", "p"]
SPEED_OF_LIGHT = 299792458.0  # m/s


def _complex_index(material: BaseMaterial, wavelength_um: float | Array) -> Array:
    if be.get_backend() == "torch" and hasattr(wavelength_um, "detach"):
        # Avoid material cache-key conversion that may require tensor->numpy bridge.
        n = material._calculate_n(wavelength_um)
        k = material._calculate_k(wavelength_um)
    else:
        n = material.n(wavelength_um)
        k = material.k(wavelength_um)
    n = be.atleast_1d(n)
    k = be.atleast_1d(k)
    return be.to_complex(n) + 1j * be.to_complex(k)


def _snell_cos(n0, theta0, n):
    """Transmitted angle cosine with forward-branch selection.

    Calculation follows 'Thin-Film Optical Filters, Fifth Edition, Macleod,
    Hugh Angus CRC Press, Ch2.6.

    Args:
        n0 (complex): Incident medium complex refractive index.
        theta0 (float): Angle of incidence in radians.
        n (complex): Medium complex refractive index.

    Returns:
        complex: Transmitted angle cosine.
    """
    nr = n.real
    k = n.imag
    return be.sqrt(nr**2 - k**2 - (n0 * be.sin(theta0)) ** 2 - 2j * nr * k) / n


def _admittance(n: complex, cos_t: complex, pol: PolSP):
    """Admittance η = sqrt(ε/μ) * n * cos(θ) for s and p polarizations.

    Calculation follows 'Thin-Film Optical Filters, Fifth Edition, Macleod,
    Hugh Angus CRC Press, Ch2.6.

    Args:
        n (complex): Medium complex refractive index.
        cos_t (complex): Cosine of the angle of propagation in the medium.
        pol (PolSP): Polarization state ('s' or 'p').

    Returns:
        complex: Optical admittance.
    """
    sqrt_eps_mu = 0.002654418729832701370374020517935  # S
    eta_s = sqrt_eps_mu * n * cos_t

    if pol == "s":
        return eta_s
    elif pol == "p":
        eta_p = sqrt_eps_mu**2 * (n.real - 1j * n.imag) ** 2 / eta_s
        return eta_p
    else:
        raise ValueError("Invalid polarization state")


def _tmm_coh(stack: ThinFilmStack, wavelength_um, theta0_rad, pol: PolSP):
    """Compute the reflection and transmission coefficients for a thin film stack.

    Calculation is vectorized over wavelength and angle of incidence.
    Based on Abelès Matrix.

    Ref:
        - Chap 13. Polarized Light and Optical Systems, Russell
          A. Chipman, Wai-Sze Tiffany Lam, and Garam Young.
        - F. Abelès, Researches sur la propagation des ondes électromagnétiques
          sinusoïdales dans les milieus stratifies. Applications aux couches
          minces, Ann. Phys. Paris, 12ième Series 5 (1950): 596–640.
        - Chap 2. Thin-Film Optical Filters, Fifth Edition, Macleod, Hugh Angus
          CRC Press.

    Args:
        stack (ThinFilmStack): The thin film stack to compute.
        wavelength_um (float | Array): Wavelength(s) in microns.
        theta0_rad (float | Array): Angle(s) of incidence in radians.
        pol (PolSP): Polarization state ('s' or 'p').

    Returns:
        tuple[Array, Array, Array, Array, Array]: (r, t, R, T, A) where:
            r: Complex reflection coefficient.
            t: Complex transmission coefficient.
            R: Reflectance.
            T: Transmittance.
            A: Absorptance.
    """
    n0 = _complex_index(stack.incident_material, wavelength_um)
    ns = _complex_index(stack.substrate_material, wavelength_um)
    cos0 = _snell_cos(n0, theta0_rad, n0)
    coss = _snell_cos(n0, theta0_rad, ns)
    eta0 = _admittance(n0, cos0, pol)
    etas = _admittance(ns, coss, pol)

    # Id initial matrix
    A = be.to_complex(be.ones_like(eta0))
    B = be.to_complex(be.zeros_like(eta0))
    C = be.to_complex(be.zeros_like(eta0))
    D = be.to_complex(be.ones_like(eta0))

    for layer in stack.layers:
        n_l = layer.n_complex(wavelength_um)
        cos_l = _snell_cos(n0, theta0_rad, n_l)
        eta_l = _admittance(n_l, cos_l, pol)
        delta = layer.phase_thickness(wavelength_um, cos_l, n_l)
        c = be.cos(delta)
        s = be.sin(delta)
        i = 1j
        mA = c
        mB = i * (s / eta_l)
        mC = i * (eta_l * s)
        mD = c
        A, B, C, D = A * mA + B * mC, A * mB + B * mD, C * mA + D * mC, C * mB + D * mD

    denom = eta0 * (A + etas * B) + C + etas * D
    denom = be.where(be.abs(denom) == 0, 1e-30 + 0j, denom)

    r = (eta0 * A + eta0 * etas * B - C - etas * D) / denom
    t = be.conj((2 * eta0) / denom)

    R = (r * be.conj(r)).real
    T = (t * be.conj(t)).real * etas.real / eta0.real
    abso = 1 - R - T
    # Absorption can also be obtained with :
    # abso = (1 - R) * (1 - etas.real / ((A + etas * B) * (C + etas * D).conj()).real)
    return r, t, R, T, abso


def _wavelength_um_to_omega(wavelength_um: float | Array) -> Array:
    """Convert wavelength in microns to angular frequency in rad/s."""
    return 2 * be.pi * SPEED_OF_LIGHT / (wavelength_um * 1e-6)


def _omega_to_wavelength_um(omega: float | Array) -> Array:
    """Convert angular frequency in rad/s to wavelength in microns."""
    return 2 * be.pi * SPEED_OF_LIGHT * 1e6 / omega


def _spectral_phase_metrics(
    stack: ThinFilmStack,
    wavelength_um,
    theta0_rad,
    pol: PolSP,
    coefficient: Literal["r", "t"] = "r",
    step_fraction: float = 1e-6,
) -> dict[str, Array]:
    """Return phase, GD and GDD of a complex coefficient with respect to ω.

    The quantities are computed from the complex reflection or transmission
    coefficient using centered finite differences in angular frequency.

    Returns:
        dict with keys ``phase``, ``gd`` and ``gdd``.
    """
    if coefficient not in ("r", "t"):
        raise ValueError("coefficient must be 'r' or 't'")

    omega0 = _wavelength_um_to_omega(wavelength_um)
    scale = be.where(be.abs(omega0) > 1.0, be.abs(omega0), 1.0)
    h = step_fraction * scale
    omega_plus = omega0 + h
    omega_minus = omega0 - h

    wl_plus = _omega_to_wavelength_um(omega_plus)
    wl_minus = _omega_to_wavelength_um(omega_minus)

    r0, t0, _, _, _ = _tmm_coh(stack, wavelength_um, theta0_rad, pol)
    r_plus, t_plus, _, _, _ = _tmm_coh(stack, wl_plus, theta0_rad, pol)
    r_minus, t_minus, _, _, _ = _tmm_coh(stack, wl_minus, theta0_rad, pol)

    q0 = r0 if coefficient == "r" else t0
    q_plus = r_plus if coefficient == "r" else t_plus
    q_minus = r_minus if coefficient == "r" else t_minus

    phase0 = be.arctan2(be.imag(q0), be.real(q0))

    delta_plus = be.arctan2(
        be.imag(q_plus * be.conj(q0)),
        be.real(q_plus * be.conj(q0)),
    )
    delta_minus = be.arctan2(
        be.imag(q0 * be.conj(q_minus)),
        be.real(q0 * be.conj(q_minus)),
    )

    gd = (delta_plus + delta_minus) / (2 * h)
    gdd = (delta_plus - delta_minus) / (h * h)

    return {"phase": phase0, "gd": gd, "gdd": gdd}


def _layer_fields(
    stack: ThinFilmStack,
    wavelength_um,
    theta0_rad,
    pol: PolSP,
    position_fraction: float | Array = 0.5,
) -> list[dict[str, Any]]:
    """Compute internal complex electric field data for each layer.

    The field is sampled at a relative position inside each layer. The result
    is a list of dictionaries, one entry per layer, containing:
    ``layer_index``, ``position_fraction``, ``E``, ``amplitude``, ``phase``,
    ``E_plus``, ``E_minus``, ``U_in`` and ``V_in``.
    """
    n0 = _complex_index(stack.incident_material, wavelength_um)
    ns = _complex_index(stack.substrate_material, wavelength_um)
    cos0 = _snell_cos(n0, theta0_rad, n0)
    coss = _snell_cos(n0, theta0_rad, ns)
    eta0 = _admittance(n0, cos0, pol)
    etas = _admittance(ns, coss, pol)

    # Global characteristic matrix M = [[A, B], [C, D]]
    A = be.to_complex(be.ones_like(eta0))
    B = be.to_complex(be.zeros_like(eta0))
    C = be.to_complex(be.zeros_like(eta0))
    D = be.to_complex(be.ones_like(eta0))

    for layer in stack.layers:
        n_l = layer.n_complex(wavelength_um)
        cos_l = _snell_cos(n0, theta0_rad, n_l)
        eta_l = _admittance(n_l, cos_l, pol)
        delta = layer.phase_thickness(wavelength_um, cos_l, n_l)
        c = be.cos(delta)
        s = be.sin(delta)
        i = 1j
        mA = c
        mB = i * (s / eta_l)
        mC = i * (eta_l * s)
        mD = c
        A, B, C, D = A * mA + B * mC, A * mB + B * mD, C * mA + D * mC, C * mB + D * mD

    denom = eta0 * (A + etas * B) + C + etas * D
    denom = be.where(be.abs(denom) == 0, 1e-30 + 0j, denom)

    r = (eta0 * A + eta0 * etas * B - C - etas * D) / denom

    layer_fields: list[dict[str, Any]] = []

    # Incident-side boundary fields for a unit-amplitude incident wave.
    U0 = 1 + r
    V0 = eta0 * (1 - r)

    # Prefix matrix from the incident side to the entrance of the current layer.
    prefix_A = be.to_complex(be.ones_like(eta0))
    prefix_B = be.to_complex(be.zeros_like(eta0))
    prefix_C = be.to_complex(be.zeros_like(eta0))
    prefix_D = be.to_complex(be.ones_like(eta0))

    for layer_index, layer in enumerate(stack.layers, start=1):
        n_l = layer.n_complex(wavelength_um)
        cos_l = _snell_cos(n0, theta0_rad, n_l)
        eta_l = _admittance(n_l, cos_l, pol)
        delta = layer.phase_thickness(wavelength_um, cos_l, n_l)

        det_prefix = prefix_A * prefix_D - prefix_B * prefix_C
        det_prefix = be.where(be.abs(det_prefix) == 0, 1e-30 + 0j, det_prefix)

        invA = prefix_D / det_prefix
        invB = -prefix_B / det_prefix
        invC = -prefix_C / det_prefix
        invD = prefix_A / det_prefix

        U_in = invA * U0 + invB * V0
        V_in = invC * U0 + invD * V0

        E_plus = 0.5 * (U_in + V_in / eta_l)
        E_minus = 0.5 * (U_in - V_in / eta_l)

        delta_z = position_fraction * delta
        E_z = E_plus * be.exp(-1j * delta_z) + E_minus * be.exp(1j * delta_z)

        layer_fields.append(
            {
                "layer_index": layer_index,
                "position_fraction": position_fraction,
                "E": E_z,
                "amplitude": be.abs(E_z),
                "phase": be.arctan2(be.imag(E_z), be.real(E_z)),
                "E_plus": E_plus,
                "E_minus": E_minus,
                "U_in": U_in,
                "V_in": V_in,
            }
        )

        c = be.cos(delta)
        s = be.sin(delta)
        i = 1j
        mA = c
        mB = i * (s / eta_l)
        mC = i * (eta_l * s)
        mD = c
        prefix_A, prefix_B, prefix_C, prefix_D = (
            prefix_A * mA + prefix_B * mC,
            prefix_A * mB + prefix_B * mD,
            prefix_C * mA + prefix_D * mC,
            prefix_C * mB + prefix_D * mD,
        )

    return layer_fields
