"""Multi-Configuration Tolerancing Module

This module tolerances a ``MultiConfiguration`` (a zoom lens, for example) as
one physical lens: a tolerance is applied with the same deviation to every
configuration, while operands and compensators can be set per configuration.
The analyses (``SensitivityAnalysis``, ``MonteCarlo``, ``InverseSensitivity``)
run on a ``MultiConfigTolerancing`` unchanged.

Kramer Harrison, 2026
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from optiland.optimization.operand import Operand
from optiland.tolerancing.compensator import QuickFocusCompensator
from optiland.tolerancing.core import Tolerancing
from optiland.tolerancing.registry import ToleranceType, tolerance_registry
from optiland.tolerancing.tolerance import Tolerance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from optiland.multiconfig import MultiConfiguration
    from optiland.optic import Optic
    from optiland.tolerancing.distributions import Distribution

_STANDARD_PICKUP_ATTRS = ("radius", "conic", "thickness")


def _pickup_driver(optic: Optic, surface_number: int, attribute: str) -> Optic | None:
    """Return the source optic of the pickup that sets an attribute.

    Pickups are applied in order, so the last matching pickup wins.

    Args:
        optic: The optic whose pickups are searched.
        surface_number: The surface of the attribute.
        attribute: A pickup ``attr_type`` or a surface attribute path, as
            returned by ``ToleranceType.pickup_targets``.

    Returns:
        The source optic, or None if no pickup sets the attribute.
    """
    driver = None
    for pickup in optic.pickups.pickups:
        if attribute in _STANDARD_PICKUP_ATTRS:
            hit = (
                pickup.attr_type == attribute
                and pickup.target_surface_idx == surface_number
            )
        else:
            path = pickup.attr_type.replace("[i]", f"[{pickup.target_surface_idx}]")
            hit = path == f"surfaces.surfaces[{surface_number}].{attribute}"
        if hit:
            driver = pickup.source_optic
    return driver


class ConfigurationGroupTolerance(ToleranceType):
    """One tolerance type applied with the same deviation to several
    configurations.

    The deviation is applied directly only to members whose attributes are
    not all driven by pickups from other members of the group. A
    pickup-driven member gets the perturbed value through its pickup, so a
    pickup with a scale and an offset gives ``scale * (v0 + delta) + offset``.
    After every ``apply`` and ``reset``, when the type writes pickup-linkable
    attributes (``pickup_targets``), the pickups of the non-master
    configurations are refreshed with ``updater.update()``. Types with no
    pickup targets (decenters, tilts, irregularity) skip the refresh: it
    would rebuild the axial surface positions of those configurations from
    their thicknesses and drop the axial part of a rigid-body move.

    Label, phase and units are those of the first member.

    Args:
        members: One tolerance type per selected configuration, in
            configuration order.
        configurations: All configurations of the multi-configuration
            system; the first is the master.

    Attributes:
        members: The tolerance types, one per selected configuration.
        direct: Whether each member receives the deviation directly.
        dependents: The non-master configurations, refreshed after changes.

    Raises:
        ValueError: If ``members`` is empty, or if an attribute of a member
            is driven by a pickup from an optic that is not a member's
            optic, such as a configuration that is not selected.
    """

    def __init__(
        self, members: Sequence[ToleranceType], configurations: Sequence[Optic]
    ) -> None:
        if not members:
            raise ValueError("A configuration group needs at least one member.")
        super().__init__(members[0].optic)
        self.members = list(members)
        self.dependents = list(configurations[1:])
        self.phase = self.members[0].phase
        self.units = self.members[0].units
        optics = [member.optic for member in self.members]
        self.direct = [
            self._is_direct(member, optics, configurations) for member in self.members
        ]
        self._refresh = any(member.pickup_targets() for member in self.members)

    @staticmethod
    def _is_direct(
        member: ToleranceType, optics: list[Optic], configurations: Sequence[Optic]
    ) -> bool:
        """Return whether a member receives the deviation directly."""
        targets = member.pickup_targets()
        driven = []
        for surface_number, attribute in targets:
            driver = _pickup_driver(member.optic, surface_number, attribute)
            if driver is None or driver is member.optic:
                driven.append(False)
            elif any(driver is optic for optic in optics):
                driven.append(True)
            else:
                index = next(
                    (i for i, c in enumerate(configurations) if c is driver), None
                )
                source = "an optic outside the multi-configuration system"
                if index is not None:
                    source = f"configuration {index}, which is not selected"
                raise ValueError(
                    f"{member.label}: {attribute} of surface {surface_number} is "
                    f"picked up from {source}. Select that configuration too."
                )
        return not (targets and all(driven))

    @property
    def label(self) -> str:
        """Return the label of the first member."""
        return self.members[0].label

    def pickup_targets(self) -> list[tuple[int, str]]:
        """Return the pickup targets of the first member."""
        return self.members[0].pickup_targets()

    def apply(self, delta: float) -> None:
        """Apply the deviation to the direct members, then refresh pickups."""
        for member, direct in zip(self.members, self.direct, strict=True):
            if direct:
                member.apply(delta)
        self._refresh_pickups()

    def reset(self) -> None:
        """Reset the direct members, then refresh pickups."""
        for member, direct in zip(
            reversed(self.members), reversed(self.direct), strict=True
        ):
            if direct:
                member.reset()
        self._refresh_pickups()

    def _refresh_pickups(self) -> None:
        if self._refresh:
            for optic in self.dependents:
                optic.updater.update()


@dataclass
class _ConfigurationOperand(Operand):
    """An operand whose name carries its configuration index."""

    configuration: int = 0

    def __str__(self) -> str:
        return f"{super().__str__()} (config {self.configuration})"


class MultiConfigTolerancing(Tolerancing):
    """A tolerancing problem for a multi-configuration system.

    The configurations are one physical lens: ``add_tolerance`` applies the
    same deviation to every selected configuration through a
    ``ConfigurationGroupTolerance``. Pickups express the design relation, so
    they win: the deviation is applied directly only to configurations whose
    attribute is not driven by a pickup from another selected configuration;
    a pickup-driven configuration gets the perturbed value through its
    pickup (``scale * (v0 + delta) + offset``). Tolerance types state their
    pickup-linkable attributes with ``ToleranceType.pickup_targets``
    (radius, fringes, conic, thickness, index and Abbe); decenters, tilts
    and irregularity are not pickup-linked and are applied to every
    configuration. Form tolerances use the clear semi-aperture of the first
    selected configuration, so every configuration sees the same surface
    error.

    ``add_default_tolerances`` and ``add_tolerances_from_drawing`` add their
    tolerances to all configurations. ``add_perturbation`` acts on
    configuration 0 only. Operands and compensators are set per
    configuration with the ``configuration`` argument.

    Configuration 0 is ``optic``, so code written for ``Tolerancing`` keeps
    working. The pickups of the other configurations are applied once when
    the problem is created and on every ``reset``.

    Args:
        multiconfig: The multi-configuration system.
        method: The method of the compensator optimizer. Defaults to
            ``"generic"``.
        tol: The tolerance of the compensator optimizer. Defaults to 1e-5.

    Attributes:
        multiconfig: The multi-configuration system.
        quick_focus_compensators: Quick focus compensators, keyed by
            configuration index.

    Raises:
        ValueError: If the configurations differ in surface count.
    """

    def __init__(
        self,
        multiconfig: MultiConfiguration,
        method: str = "generic",
        tol: float = 1e-5,
    ) -> None:
        configurations = multiconfig.configurations
        counts = [optic.surfaces.num_surfaces for optic in configurations]
        if len(set(counts)) > 1:
            raise ValueError(
                "All configurations must have the same number of surfaces for "
                f"tolerancing; got {counts}."
            )
        super().__init__(configurations[0], method=method, tol=tol)
        self.multiconfig = multiconfig
        self.quick_focus_compensators: dict[int, QuickFocusCompensator] = {}
        self._update_dependents()

    @property
    def configurations(self) -> list[Optic]:
        """Return the configurations of the multi-configuration system."""
        return self.multiconfig.configurations

    def _select(self, configurations: int | Sequence[int] | str) -> list[int]:
        """Return sorted configuration indices for a selection argument.

        Raises:
            ValueError: If the selection is empty, repeats an index, or names
                a configuration that does not exist.
        """
        n = len(self.configurations)
        if configurations == "all":
            return list(range(n))
        if isinstance(configurations, int):
            configurations = [configurations]
        indices = list(configurations)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError(
                f"Select at least one configuration, without repeats; got {indices}."
            )
        for index in indices:
            if not isinstance(index, int) or not 0 <= index < n:
                raise ValueError(
                    f"Configuration {index!r} does not exist; there are {n}."
                )
        return sorted(indices)

    def _update_dependents(self) -> None:
        """Apply the pickups of the non-master configurations."""
        for optic in self.configurations[1:]:
            optic.updater.update()

    def add_operand(
        self,
        operand_type: str,
        input_data: dict = None,
        target: float = None,
        weight: float = 1.0,
        min_val: float = None,
        max_val: float = None,
        *,
        configuration: int | Literal["all"] = 0,
    ) -> None:
        """Add an operand evaluated on one or every configuration.

        Args:
            operand_type: The type of the operand.
            input_data: Additional input data for the operand. Its
                ``"optic"`` is set to the configuration's optic.
            target: The target value (equality operand). If it and both
                bounds are omitted, the current value of each configuration
                is captured as its target.
            weight: The weight of the operand.
            min_val: Lower bound (inequality operand).
            max_val: Upper bound (inequality operand).
            configuration: The configuration index, or ``"all"`` to add one
                operand per configuration. With ``"all"`` the operand names
                (result columns) end with ``" (config i)"``.

        Raises:
            ValueError: If the configuration does not exist, or if
                ``input_data`` names an optic other than the configuration's.
        """
        indices = self._select(configuration)
        for index in indices:
            optic = self.configurations[index]
            data = dict(input_data or {})
            if data.get("optic", optic) is not optic:
                raise ValueError(
                    "input_data['optic'] differs from the selected configuration; "
                    "omit it and use the configuration argument."
                )
            data["optic"] = optic
            if configuration == "all":
                operand = _ConfigurationOperand(
                    operand_type, target, min_val, max_val, weight, data, index
                )
            else:
                operand = Operand(operand_type, target, min_val, max_val, weight, data)
            self.operands.append(operand)

    def add_tolerance(
        self,
        tolerance_type: str,
        limit: float | None = None,
        *,
        minus: float | None = None,
        plus: float | None = None,
        distribution: str | Distribution = "normal",
        configurations: Sequence[int] | Literal["all"] = "all",
        **kwargs,
    ) -> Tolerance:
        """Add a tolerance applied with the same deviation to configurations.

        Args:
            tolerance_type: The registered tolerance type, as in
                ``Tolerancing.add_tolerance``.
            limit: Symmetric limit; the tolerance spans ``[-limit, +limit]``.
            minus: Lower limit, for asymmetric tolerances. Requires ``plus``.
            plus: Upper limit, for asymmetric tolerances. Requires ``minus``.
            distribution: The distribution used by Monte Carlo analyses.
            configurations: The configuration indices, or ``"all"``
                (default).
            **kwargs: Arguments for the tolerance type.

        Returns:
            The new tolerance, whose type is a ``ConfigurationGroupTolerance``.

        Raises:
            ValueError: If the limits are invalid, the tolerance type is
                unknown, a configuration does not exist, or a selected
                attribute is picked up from a configuration that is not
                selected.
        """
        minus, plus = self._tolerance_limits(limit, minus, plus)
        tolerance_cls = tolerance_registry.get(tolerance_type)
        members = []
        for index in self._select(configurations):
            member_kwargs = kwargs
            if members and self._shares_semi_aperture(members[0], kwargs):
                member_kwargs = {**kwargs, "semi_aperture": members[0].semi_aperture}
            members.append(tolerance_cls(self.configurations[index], **member_kwargs))
        group = ConfigurationGroupTolerance(members, self.configurations)
        tolerance = Tolerance(group, minus, plus, distribution)
        self.perturbations.append(tolerance)
        return tolerance

    @staticmethod
    def _shares_semi_aperture(first: ToleranceType, kwargs: dict) -> bool:
        """Return whether the first member's semi-aperture is passed on."""
        parameters = inspect.signature(type(first)).parameters
        return (
            kwargs.get("semi_aperture") is None
            and "semi_aperture" in parameters
            and hasattr(first, "semi_aperture")
        )

    def add_compensator(
        self,
        variable_type: str,
        *,
        configuration: int | Literal["all"] = 0,
        **kwargs,
    ) -> None:
        """Add a compensator on one or every configuration.

        Args:
            variable_type: The variable type, such as ``"thickness"``, or
                ``"quick_focus"`` to move the image surface of the
                configuration to best focus on every trial (see
                ``QuickFocusCompensator``). Quick focus runs before the
                optimizer compensators; adding it again for a configuration
                replaces it. Its result column ends with ``" (config i)"``
                for configurations other than 0. A variable on an attribute
                that is picked up from another configuration is overwritten
                by the pickup; unlink it first to compensate on its own.
            configuration: The configuration index, or ``"all"`` to add one
                compensator per configuration.
            **kwargs: Arguments for the variable or the quick focus.

        Raises:
            ValueError: If the configuration does not exist.
        """
        for index in self._select(configuration):
            optic = self.configurations[index]
            if variable_type != "quick_focus":
                self.compensator.add_variable(optic, variable_type, **kwargs)
                continue
            quick_focus = QuickFocusCompensator(optic, **kwargs)
            if index != 0:
                quick_focus.label = f"{QuickFocusCompensator.label} (config {index})"
            self.quick_focus_compensators[index] = quick_focus

    def apply_compensators(self) -> dict[str, float]:
        """Apply the quick focus compensators, then the optimizer.

        Returns:
            The compensator values, keyed by result column name.
        """
        result = {}
        for index in sorted(self.quick_focus_compensators):
            result.update(self.quick_focus_compensators[index].apply())
        result.update(super().apply_compensators())
        return result

    def reset(self) -> None:
        """Reset every configuration to its nominal state."""
        # quick focus first, as in Tolerancing.reset
        for index in sorted(self.quick_focus_compensators, reverse=True):
            self.quick_focus_compensators[index].reset()
        super().reset()
        # compensator variables may drive pickups of other configurations
        self._update_dependents()
