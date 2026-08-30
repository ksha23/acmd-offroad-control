"""Protocol contracts for the single-operator hardware-in-the-loop schedule.

A within-subject schedule can be confounded by its own ordering, so these
tests establish that the generated plan is balanced across filters and
scenarios, that filter precedence is near-balanced within each scenario, that
the ordering is reproducible from its recorded seed, and that operator
identifiers stay anonymised.
"""

from argparse import Namespace
from collections import Counter

from benchmarking.human_delay_compensation_rounds import (
    build_round_conditions,
    validate_args,
)


def _canonical_args(**overrides) -> Namespace:
    values = {
        "study_id": "acmd_single_operator_hil",
        "operator_id": "op01",
        "filters": ["none", "dob_cbf"],
        "delays": [0.0],
        "convoy": ["lead_brake", "cut_in", "stalled"],
        "terrains": ["clay"],
        "paths": ["straight"],
        "speeds": [4.0],
        "bumpiness": [0],
        "rounds": 5,
        "base_seed": 910,
        "order": "randomized-blocks",
        "order_seed": 20260719,
        "practice_rounds": 3,
    }
    values.update(overrides)
    return Namespace(**values)


def _pair_key(cell) -> tuple:
    return (
        cell.delay, cell.terrain, cell.path, cell.speed, cell.bumpiness,
        cell.seed, cell.convoy, cell.repetition,
    )


def test_canonical_single_operator_plan_is_balanced_and_reproducible():
    args = _canonical_args()
    validate_args(args)
    plan = build_round_conditions(args)
    assert plan == build_round_conditions(args)
    assert len(plan) == 30

    for block_number in range(1, 6):
        block = [cell for cell in plan if cell.block == block_number]
        assert len(block) == 6
        assert Counter(cell.filter_name for cell in block) == {
            "none": 3, "dob_cbf": 3,
        }
        assert Counter(cell.convoy for cell in block) == {
            "lead_brake": 2, "cut_in": 2, "stalled": 2,
        }
        assert all(_pair_key(block[i]) != _pair_key(block[i + 1])
                   for i in range(len(block) - 1))

    filters = [cell.filter_name for cell in plan]
    assert all(not (filters[i] == filters[i + 1] == filters[i + 2])
               for i in range(len(filters) - 2))


def test_filter_precedence_is_nearly_balanced_for_each_scenario():
    plan = build_round_conditions(_canonical_args())
    for scenario in ("lead_brake", "cut_in", "stalled"):
        first_filters = []
        for block_number in range(1, 6):
            pair = [cell for cell in plan
                    if cell.block == block_number and cell.convoy == scenario]
            first_filters.append(pair[0].filter_name)
        assert sorted(Counter(first_filters).values()) == [2, 3]


def test_anonymized_identifiers_reject_spaces_and_names_with_slashes():
    for bad_id in ("operator one", "../name", ""):
        args = _canonical_args(operator_id=bad_id)
        try:
            validate_args(args)
        except SystemExit:
            pass
        else:  # pragma: no cover - makes the failed value obvious
            raise AssertionError(f"accepted unsafe operator id: {bad_id!r}")
