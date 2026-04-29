"""Uneven distribution: detect a row of similar nodes and shift the last one."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

from evals.common import MODE_DEFINITIONS, normalize_spec
from evals.corrupters.base import Corrupter, Corruption


WIDTH_TOLERANCE = 0.15      # widths within 15% of mean
Y_TOLERANCE = 30            # top edges within 30 px
GAP_CV_THRESHOLD = 0.35     # coefficient of variation for inter-element gaps


def _coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    if m == 0:
        return 0.0
    return pstdev(values) / abs(m)


def _detect_row(spec_dict: dict[str, Any]) -> list[int] | None:
    nodes = spec_dict.get("nodes", [])
    indexed = [(i, n) for i, n in enumerate(nodes)]
    # Bucket by node type; consider rows within a single type.
    by_type: dict[str, list[tuple[int, dict]]] = {}
    for i, n in indexed:
        by_type.setdefault(n.get("type", ""), []).append((i, n))

    for type_name, group in by_type.items():
        if len(group) < 3:
            continue
        # Cluster items with similar y-coordinates (within Y_TOLERANCE).
        sorted_by_y = sorted(group, key=lambda kv: int(kv[1]["y"]))
        i = 0
        while i < len(sorted_by_y) - 2:
            base_y = int(sorted_by_y[i][1]["y"])
            row = [sorted_by_y[i]]
            j = i + 1
            while j < len(sorted_by_y):
                if abs(int(sorted_by_y[j][1]["y"]) - base_y) <= Y_TOLERANCE:
                    row.append(sorted_by_y[j])
                    j += 1
                else:
                    break
            if len(row) >= 3:
                # Order by x.
                row.sort(key=lambda kv: int(kv[1]["x"]))
                widths = [int(n["width"]) for _, n in row]
                if max(widths) - min(widths) <= mean(widths) * WIDTH_TOLERANCE * 2:
                    # Compute gaps between adjacent right-edges.
                    gaps = []
                    for k in range(len(row) - 1):
                        a = row[k][1]
                        b = row[k + 1][1]
                        gap = int(b["x"]) - (int(a["x"]) + int(a["width"]))
                        gaps.append(gap)
                    if all(g >= 0 for g in gaps) and _coefficient_of_variation(gaps) < GAP_CV_THRESHOLD:
                        return [idx for idx, _ in row]
            i = j if j > i + 1 else i + 1
    return None


def _apply(spec_dict: dict[str, Any]) -> Corruption | None:
    spec = normalize_spec(spec_dict)
    row_indices = _detect_row(spec)
    if not row_indices:
        return None

    # Shift the rightmost item in the row right by 1.5x the natural gap.
    nodes = spec["nodes"]
    row = [nodes[i] for i in row_indices]
    row.sort(key=lambda n: int(n["x"]))
    gaps = []
    for k in range(len(row) - 1):
        gap = int(row[k + 1]["x"]) - (int(row[k]["x"]) + int(row[k]["width"]))
        gaps.append(gap)
    natural_gap = int(mean(gaps)) if gaps else 50
    shift = max(60, int(natural_gap * 1.5))

    last_node_in_row = row[-1]
    last_idx = next(i for i in row_indices if nodes[i] is last_node_in_row)
    canvas_w = int(spec.get("canvas_width", 800))
    old_x = int(last_node_in_row["x"])
    new_x = min(canvas_w - int(last_node_in_row["width"]), old_x + shift)
    if new_x == old_x:
        return None  # canvas couldn't accommodate the shift
    spec["nodes"][last_idx] = {**last_node_in_row, "x": new_x}
    return Corruption(
        spec=spec,
        description=(
            f"row of {len(row_indices)} {last_node_in_row.get('type')} nodes — "
            f"shifted node[{last_idx}].x: {old_x} -> {new_x}"
        ),
        changed_node_indices=[last_idx],
    )


_DEF = MODE_DEFINITIONS["uneven_distribution"]
CORRUPTER = Corrupter(
    id="uneven_distribution",
    name=_DEF["name"],
    description=_DEF["description"],
    apply=_apply,
)
