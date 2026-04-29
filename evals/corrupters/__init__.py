"""Registry of real-design corrupters. semiotic_mismatch is intentionally
omitted — most real specs lack swappable icons, see plan."""

from evals.corrupters.base import Corrupter, Corruption
from evals.corrupters.crowding import CORRUPTER as crowding
from evals.corrupters.inconsistency import CORRUPTER as inconsistency
from evals.corrupters.misalignment import CORRUPTER as misalignment
from evals.corrupters.nonsensical_hierarchy import CORRUPTER as nonsensical_hierarchy
from evals.corrupters.overflow import CORRUPTER as overflow
from evals.corrupters.overlap import CORRUPTER as overlap
from evals.corrupters.poor_contrast import CORRUPTER as poor_contrast
from evals.corrupters.uneven_distribution import CORRUPTER as uneven_distribution

CORRUPTERS: list[Corrupter] = [
    overflow,
    poor_contrast,
    overlap,
    misalignment,
    nonsensical_hierarchy,
    inconsistency,
    uneven_distribution,
    crowding,
]

__all__ = ["CORRUPTERS", "Corrupter", "Corruption"]
