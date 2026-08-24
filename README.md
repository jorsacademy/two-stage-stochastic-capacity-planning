# Two-Stage Stochastic Capacity Planning

A compact Operations Research project for capacity planning under uncertain demand.

The model is formulated as a **two-stage stochastic mixed-integer linear program (MILP)** and solved in Python with `scipy.optimize.milp`, which uses the HiGHS optimization backend.

All data in this repository is synthetic.

## Problem

A planner must decide how much capacity to install **before** demand is known.

After one of several demand scenarios is realized, the planner can:

- use installed internal capacity,
- outsource part of the demand,
- leave demand unmet at a high penalty.

The first-stage capacity decisions are shared across all scenarios. This is the non-anticipativity requirement: the planner cannot choose a different capacity investment after learning which future scenario occurs.

## Mathematical formulation

Sets:

- `K`: capacity types
- `S`: demand scenarios

First-stage decision:

- `x_k`: integer number of installed units of capacity type `k`

Second-stage decisions for each scenario `s`:

- `y_ks`: internal output from capacity type `k`
- `o_s`: outsourced quantity
- `u_s`: unmet demand

Objective:

```text
minimize

sum_k investment_cost_k * x_k

+ sum_s probability_s * (
      sum_k operating_cost_k * y_ks
    + outsource_cost_s * o_s
    + unmet_penalty_s * u_s
  )
```

Internal capacity constraints:

```text
y_ks <= unit_capacity_k * x_k
```

Demand balance:

```text
sum_k y_ks + o_s + u_s >= demand_s
```

Outsourcing limits:

```text
o_s <= outsource_limit_s
```

The capacity variables `x_k` appear only once in the deterministic equivalent and are shared by every scenario. This directly enforces non-anticipativity.

## Stochastic-programming metrics

The project computes three benchmark models.

### Recourse Problem (RP)

The full two-stage stochastic optimization model.

### Expected-Value Solution (EEV)

Uncertain demand and recourse parameters are replaced by their probability-weighted expectations. The resulting first-stage plan is then fixed and evaluated against the original scenarios.

### Wait-and-See (WS)

Each scenario is solved independently as if perfect information about future demand were available before the capacity decision.

From these models:

```text
VSS = EEV - RP
EVPI = RP - WS
```

for this cost-minimization problem.

- **VSS** measures the value of solving the stochastic model instead of planning only for expected conditions.
- **EVPI** measures the maximum value of perfect information about the future.

## Synthetic instance

The sample includes:

- two capacity technologies,
- three demand scenarios,
- scenario probabilities,
- scenario-dependent outsourcing costs and limits,
- scenario-dependent unmet-demand penalties.

The demand scenarios are `LOW`, `MEDIUM`, and `HIGH`. These labels are purely descriptive and do not refer to any organization.

## Verified sample result

The included instance solves to:

```text
Optimal first-stage capacity:
BASE = 3 units
FLEX = 0 units

RP   = 4538.125
EEV  = 4982.500
WS   = 3787.500

VSS  = 444.375
EVPI = 750.625
```

The ordering is therefore:

```text
WS <= RP <= EEV
```

as expected for this minimization problem.

## Project structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src
│   ├── __init__.py
│   ├── model.py
│   └── run.py
└── tests
    └── test_model.py
```

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m src.run
```

## Test

```bash
pytest -q
```

The tests verify:

- scenario probabilities,
- expected-demand construction,
- capacity bounds,
- recourse feasibility,
- outsourcing limits,
- demand satisfaction,
- stochastic-value ordering,
- deterministic regression values,
- invalid probability handling.

## Solver

This project uses SciPy MILP with HiGHS. No commercial solver is required.

## License

Non-commercial use only. Commercial use is prohibited without separate written permission. See `LICENSE`.
