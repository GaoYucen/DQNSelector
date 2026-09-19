from dqnselector.journal_protocol import BUDGETS, instance_world_seed, scenario_rank_key


def _rows(ec):
    rows = []
    for budget in BUDGETS:
        rows.extend([
            {"method": "DQNSelector-J", "k": budget, "ec": ec},
            {"method": "DegGreedy", "k": budget, "ec": ec - .1},
            {"method": "PIANO", "k": budget, "ec": ec - .05},
        ])
    return rows


def test_instance_world_seeds_are_stable_and_purpose_separated():
    assert instance_world_seed("a" * 64, "evaluation-2000") == instance_world_seed("a" * 64, "evaluation-2000")
    assert instance_world_seed("a" * 64, "evaluation-2000") != instance_world_seed("a" * 64, "validation-500")


def test_scenario_degeneracy_is_assessed_over_development_instances():
    saturated = _rows(.99)
    useful = _rows(.5)
    # A single saturated instance does not discard a scenario whose pooled
    # development distribution is otherwise usable.
    assert scenario_rank_key({"gowalla": [saturated, useful], "brightkite": [useful, useful]})[0] >= 0
