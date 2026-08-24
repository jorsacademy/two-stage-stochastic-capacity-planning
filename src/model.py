from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class CapacityType:
    name: str
    unit_capacity: float
    investment_cost: float
    operating_cost: float
    max_units: int


@dataclass(frozen=True)
class Scenario:
    name: str
    probability: float
    demand: float
    outsource_cost: float
    outsource_limit: float
    unmet_penalty: float


@dataclass(frozen=True)
class ModelData:
    capacity_types: Tuple[CapacityType, ...]
    scenarios: Tuple[Scenario, ...]


@dataclass
class PlanResult:
    objective: float
    capacities: Dict[str, int]
    internal: Dict[Tuple[str, str], float]
    outsourced: Dict[str, float]
    unmet: Dict[str, float]
    investment_cost: float
    expected_recourse_cost: float


@dataclass
class StochasticMetrics:
    recourse_problem: PlanResult
    expected_value_plan: PlanResult
    eev_objective: float
    wait_and_see_objective: float
    vss: float
    evpi: float


def sample_data() -> ModelData:
    """Return a deterministic synthetic two-stage capacity-planning instance."""
    capacity_types = (
        CapacityType("BASE", unit_capacity=55.0, investment_cost=920.0,
                     operating_cost=7.5, max_units=4),
        CapacityType("FLEX", unit_capacity=30.0, investment_cost=620.0,
                     operating_cost=5.0, max_units=4),
    )
    scenarios = (
        Scenario("LOW", probability=0.25, demand=100.0,
                 outsource_cost=18.0, outsource_limit=40.0, unmet_penalty=75.0),
        Scenario("MEDIUM", probability=0.50, demand=155.0,
                 outsource_cost=23.0, outsource_limit=45.0, unmet_penalty=90.0),
        Scenario("HIGH", probability=0.25, demand=225.0,
                 outsource_cost=32.0, outsource_limit=50.0, unmet_penalty=120.0),
    )
    return ModelData(capacity_types=capacity_types, scenarios=scenarios)


def _validate_data(data: ModelData) -> None:
    if not data.capacity_types:
        raise ValueError("At least one capacity type is required.")
    if not data.scenarios:
        raise ValueError("At least one scenario is required.")

    probability_sum = sum(s.probability for s in data.scenarios)
    if not np.isclose(probability_sum, 1.0, atol=1e-9):
        raise ValueError("Scenario probabilities must sum to 1.")

    names = [c.name for c in data.capacity_types]
    if len(names) != len(set(names)):
        raise ValueError("Capacity type names must be unique.")

    scenario_names = [s.name for s in data.scenarios]
    if len(scenario_names) != len(set(scenario_names)):
        raise ValueError("Scenario names must be unique.")

    for c in data.capacity_types:
        if c.unit_capacity <= 0 or c.investment_cost < 0 or c.operating_cost < 0:
            raise ValueError(f"Invalid capacity parameters for {c.name}.")
        if c.max_units < 0:
            raise ValueError(f"max_units must be non-negative for {c.name}.")

    for s in data.scenarios:
        if s.probability < 0 or s.demand < 0:
            raise ValueError(f"Invalid probability or demand in scenario {s.name}.")
        if s.outsource_cost < 0 or s.outsource_limit < 0 or s.unmet_penalty < 0:
            raise ValueError(f"Invalid recourse parameters in scenario {s.name}.")


def solve_two_stage(
    data: ModelData,
    fixed_capacities: Mapping[str, int] | None = None,
) -> PlanResult:
    """
    Solve the deterministic equivalent of a two-stage stochastic MILP.

    First-stage decisions:
        x_k = integer number of capacity units of type k.

    Second-stage decisions for each scenario s:
        y_ks = internal output from capacity type k,
        o_s  = outsourced quantity,
        u_s  = unmet demand.

    The same x_k variables are shared across all scenarios, which enforces
    non-anticipativity.
    """
    _validate_data(data)

    capacity_types = list(data.capacity_types)
    scenarios = list(data.scenarios)

    if fixed_capacities is not None:
        unknown = set(fixed_capacities) - {c.name for c in capacity_types}
        if unknown:
            raise ValueError(f"Unknown capacity types in fixed_capacities: {sorted(unknown)}")

    x_idx: Dict[int, int] = {}
    y_idx: Dict[Tuple[int, int], int] = {}
    o_idx: Dict[int, int] = {}
    u_idx: Dict[int, int] = {}

    n_vars = 0
    for k in range(len(capacity_types)):
        x_idx[k] = n_vars
        n_vars += 1

    for s in range(len(scenarios)):
        for k in range(len(capacity_types)):
            y_idx[k, s] = n_vars
            n_vars += 1
        o_idx[s] = n_vars
        n_vars += 1
        u_idx[s] = n_vars
        n_vars += 1

    c = np.zeros(n_vars)
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)
    integrality = np.zeros(n_vars, dtype=int)

    for k, cap in enumerate(capacity_types):
        c[x_idx[k]] = cap.investment_cost
        integrality[x_idx[k]] = 1
        ub[x_idx[k]] = float(cap.max_units)

        if fixed_capacities is not None:
            value = int(fixed_capacities.get(cap.name, 0))
            if value < 0 or value > cap.max_units:
                raise ValueError(f"Fixed capacity for {cap.name} is out of bounds.")
            lb[x_idx[k]] = float(value)
            ub[x_idx[k]] = float(value)

    for s, scenario in enumerate(scenarios):
        p = scenario.probability
        for k, cap in enumerate(capacity_types):
            c[y_idx[k, s]] = p * cap.operating_cost
        c[o_idx[s]] = p * scenario.outsource_cost
        c[u_idx[s]] = p * scenario.unmet_penalty
        ub[o_idx[s]] = scenario.outsource_limit

    rows: List[Dict[int, float]] = []
    row_lb: List[float] = []
    row_ub: List[float] = []

    def add_row(coeffs: Dict[int, float], lower: float, upper: float) -> None:
        rows.append(coeffs)
        row_lb.append(lower)
        row_ub.append(upper)

    # Internal production cannot exceed installed first-stage capacity.
    for s in range(len(scenarios)):
        for k, cap in enumerate(capacity_types):
            add_row(
                {y_idx[k, s]: 1.0, x_idx[k]: -cap.unit_capacity},
                -np.inf,
                0.0,
            )

    # Demand balance for every scenario.
    for s, scenario in enumerate(scenarios):
        coeffs = {y_idx[k, s]: 1.0 for k in range(len(capacity_types))}
        coeffs[o_idx[s]] = 1.0
        coeffs[u_idx[s]] = 1.0
        add_row(coeffs, scenario.demand, np.inf)

    matrix = lil_matrix((len(rows), n_vars), dtype=float)
    for r, coeffs in enumerate(rows):
        for col, value in coeffs.items():
            matrix[r, col] = value

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(
            matrix.tocsr(),
            np.asarray(row_lb, dtype=float),
            np.asarray(row_ub, dtype=float),
        ),
        options={"disp": False},
    )

    if not result.success or result.x is None:
        raise RuntimeError(f"Optimization failed: {result.message}")

    values = result.x
    capacities = {
        cap.name: int(round(values[x_idx[k]]))
        for k, cap in enumerate(capacity_types)
    }
    internal = {
        (cap.name, scenario.name): float(values[y_idx[k, s]])
        for s, scenario in enumerate(scenarios)
        for k, cap in enumerate(capacity_types)
    }
    outsourced = {
        scenario.name: float(values[o_idx[s]])
        for s, scenario in enumerate(scenarios)
    }
    unmet = {
        scenario.name: float(values[u_idx[s]])
        for s, scenario in enumerate(scenarios)
    }

    investment_cost = sum(
        capacities[cap.name] * cap.investment_cost for cap in capacity_types
    )
    expected_recourse_cost = float(result.fun) - investment_cost

    return PlanResult(
        objective=float(result.fun),
        capacities=capacities,
        internal=internal,
        outsourced=outsourced,
        unmet=unmet,
        investment_cost=float(investment_cost),
        expected_recourse_cost=float(expected_recourse_cost),
    )


def expected_value_data(data: ModelData) -> ModelData:
    """Collapse uncertainty into one probability-one expected-demand scenario."""
    _validate_data(data)

    expected_demand = sum(s.probability * s.demand for s in data.scenarios)
    expected_outsource_cost = sum(
        s.probability * s.outsource_cost for s in data.scenarios
    )
    expected_outsource_limit = sum(
        s.probability * s.outsource_limit for s in data.scenarios
    )
    expected_unmet_penalty = sum(
        s.probability * s.unmet_penalty for s in data.scenarios
    )

    expected_scenario = Scenario(
        name="EXPECTED",
        probability=1.0,
        demand=expected_demand,
        outsource_cost=expected_outsource_cost,
        outsource_limit=expected_outsource_limit,
        unmet_penalty=expected_unmet_penalty,
    )
    return ModelData(
        capacity_types=data.capacity_types,
        scenarios=(expected_scenario,),
    )


def wait_and_see_objective(data: ModelData) -> float:
    """
    Compute the perfect-information benchmark.

    Each scenario is solved independently with its own first-stage capacity
    decision, then scenario-optimal costs are probability weighted.
    """
    _validate_data(data)

    total = 0.0
    for scenario in data.scenarios:
        one_scenario = Scenario(
            name=scenario.name,
            probability=1.0,
            demand=scenario.demand,
            outsource_cost=scenario.outsource_cost,
            outsource_limit=scenario.outsource_limit,
            unmet_penalty=scenario.unmet_penalty,
        )
        result = solve_two_stage(
            ModelData(data.capacity_types, (one_scenario,))
        )
        total += scenario.probability * result.objective
    return float(total)


def stochastic_metrics(data: ModelData) -> StochasticMetrics:
    """Solve RP, EEV and WS benchmarks and return VSS and EVPI."""
    rp = solve_two_stage(data)

    ev_plan = solve_two_stage(expected_value_data(data))
    eev = solve_two_stage(data, fixed_capacities=ev_plan.capacities)

    ws = wait_and_see_objective(data)

    vss = eev.objective - rp.objective
    evpi = rp.objective - ws

    # Numerical noise can create tiny negative values near zero.
    if vss < -1e-7 or evpi < -1e-7:
        raise RuntimeError("Stochastic value metrics violated expected ordering.")

    return StochasticMetrics(
        recourse_problem=rp,
        expected_value_plan=ev_plan,
        eev_objective=float(eev.objective),
        wait_and_see_objective=float(ws),
        vss=float(max(0.0, vss)),
        evpi=float(max(0.0, evpi)),
    )
