"""Display actual bundle totals rather than ambiguous distribution parameters."""

from __future__ import annotations

from optiland_gui.services.layout_rays import normalize_ray_count, ray_counts

from .spinbox_hit_areas import SpinBox


class RayCountSpinBox(SpinBox):
    """Step through complete bundles while displaying the actual ray total."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.distribution = "line_y"
        self.set_distribution(self.distribution)
        self.editingFinished.connect(self.normalize)

    def set_distribution(self, distribution):
        self.distribution = distribution
        counts = ray_counts(distribution)
        self.setRange(counts[0], counts[-1])
        self.setToolTip(
            "Total rays per field, including the chief ray. "
            "Line layouts use odd totals; hexapolar uses complete hexagonal bundles. "
            "Entered values round up to the next supported total."
        )
        self.normalize()

    def normalize(self):
        self.setValue(normalize_ray_count(self.value(), self.distribution))

    def stepBy(self, steps):  # noqa: N802
        self.normalize()
        counts = ray_counts(self.distribution)
        index = max(0, min(len(counts) - 1, counts.index(self.value()) + steps))
        self.setValue(counts[index])
