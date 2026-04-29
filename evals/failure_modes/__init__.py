"""Registry of all failure-mode generators."""

from evals.failure_modes.base import FailureMode, Variant
from evals.failure_modes.crowding import MODE as crowding
from evals.failure_modes.inconsistency import MODE as inconsistency
from evals.failure_modes.misalignment import MODE as misalignment
from evals.failure_modes.nonsensical_hierarchy import MODE as nonsensical_hierarchy
from evals.failure_modes.overflow import MODE as overflow
from evals.failure_modes.overlap import MODE as overlap
from evals.failure_modes.poor_contrast import MODE as poor_contrast
from evals.failure_modes.semiotic_mismatch import MODE as semiotic_mismatch
from evals.failure_modes.uneven_distribution import MODE as uneven_distribution

FAILURE_MODES: list[FailureMode] = [
    uneven_distribution,
    misalignment,
    inconsistency,
    nonsensical_hierarchy,
    crowding,
    semiotic_mismatch,
    overflow,
    poor_contrast,
    overlap,
]

__all__ = ["FAILURE_MODES", "FailureMode", "Variant"]
