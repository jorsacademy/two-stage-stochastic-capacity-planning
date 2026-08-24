import pytest

from src.model import (
    expected_value_data,
    sample_data,
    solve_two_stage,
    stochastic_metrics,
)


def test_probabilities_and_expected_demand():
    data = sample_data()
    assert sum(s.probability for s in data.scenarios) == pytest.approx(1.0)

    ev = expected_value_data(data)
    expected = sum(s.probability * s.demand for s in data.scenarios)
    assert ev.scenarios[0].demand == pytest.approx(expected)


def test_recourse_solution_is_feasible():
    data = sample_data()
    result = solve_two_stage(data)

    for cap in data.capacity_types:
        assert 0 <= result.capacities[cap.name] <= cap.max_units

    for scenario in data.scenarios:
        internal_total = 0.0
        for cap in data.capacity_types:
            output = result.internal[cap.name, scenario.name]
            assert output >= -1e-8
            assert output <= (
                cap.unit_capacity * result.capacities[cap.name] + 1e-7
            )
            internal_total += output

        outsourced = result.outsourced[scenario.name]
        unmet = result.unmet[scenario.name]

        assert 0 <= outsourced <= scenario.outsource_limit + 1e-7
        assert unmet >= -1e-8
        assert internal_total + outsourced + unmet >= scenario.demand - 1e-7


def test_stochastic_value_ordering_and_regression():
    data = sample_data()
    metrics = stochastic_metrics(data)

    assert metrics.wait_and_see_objective <= metrics.recourse_problem.objective + 1e-7
    assert metrics.recourse_problem.objective <= metrics.eev_objective + 1e-7
    assert metrics.vss >= -1e-8
    assert metrics.evpi >= -1e-8

    assert metrics.recourse_problem.capacities == {"BASE": 3, "FLEX": 0}
    assert metrics.recourse_problem.objective == pytest.approx(4538.125, abs=1e-6)
    assert metrics.eev_objective == pytest.approx(4982.5, abs=1e-6)
    assert metrics.wait_and_see_objective == pytest.approx(3787.5, abs=1e-6)
    assert metrics.vss == pytest.approx(444.375, abs=1e-6)
    assert metrics.evpi == pytest.approx(750.625, abs=1e-6)


def test_invalid_probability_sum_is_rejected():
    data = sample_data()
    bad = type(data)(
        capacity_types=data.capacity_types,
        scenarios=(data.scenarios[0],),
    )
    with pytest.raises(ValueError, match="probabilities must sum to 1"):
        solve_two_stage(bad)
