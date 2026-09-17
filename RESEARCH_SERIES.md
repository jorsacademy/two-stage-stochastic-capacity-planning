# Stochastic, Robust, and Risk-Aware Optimization Research Series

This file maps repositories that study decisions under uncertainty. It is an index only: every repository remains independent because the uncertainty model, information structure, risk measure, or downstream decision problem differs.

## Classical stochastic programming

- `two-stage-stochastic-capacity-planning` — two-stage stochastic MILP with non-anticipativity, recourse, VSS, and EVPI.
- `stochastic-cvrp-sample-average-approximation-python` — SAA for uncertain routing rather than capacity planning.
- `stochastic-project-selection-optimization-pulp` — scenario-based project selection.
- `airline-operations-under-uncertainty-stochastic-optimization` — airline operations under uncertainty.
- `parallel-monte-carlo-stochastic-optimization-python` — Monte Carlo based stochastic evaluation and optimization.
- `generative-supply-chain-scenarios-stochastic-optimization-pytorch` — learned scenario generation connected to stochastic optimization.

## Multistage stochastic optimization

- `sddp-multistage-energy-storage` — multistage stochastic dynamic programming/decomposition for energy storage.
- `mpi-sppy-multistage-stochastic-planning` — distributed multistage stochastic programming.
- `approximate-dynamic-programming-fleet-inventory` — sequential approximate dynamic programming; cross-listed with the sequential-decision series because it optimizes policies rather than one deterministic-equivalent model.

## Chance-constrained and robust optimization

- `chance-constrained-inventory-optimization-python` — explicit probabilistic service constraints.
- `robust-supply-chain-network-optimization` — uncertainty-set based robust network design.
- `robust-healthcare-inventory-optimization` — robust inventory decisions in a healthcare setting.
- `risk-aware-convoy-escort-allocation-milp` — explicit risk-aware resource allocation.
- `risk-based-resource-allocation-milp-python` — risk-weighted allocation model.

## Distributionally robust optimization

- `wasserstein-dro-inventory-optimization-python` — Wasserstein ambiguity sets for inventory.
- `distributionally-robust-supply-chain-optimization` — DRO at network/supply-chain level.
- `distributionally-robust-decision-focused-learning` — combines DRO with decision-focused learning and therefore also belongs to the decision-focused series.

## Prediction uncertainty connected to optimization

- `conformal-prediction-robust-inventory-optimization-python` — conformal predictive uncertainty converted into a robust decision model.
- `demand-forecasting-plus-inventory-control` — forecast-then-control pipeline; useful as a baseline contrast to decision-focused or robust approaches.
- `contextual-optimization-newsvendor` — context-conditioned operational decisions rather than a fixed unconditional uncertainty model.

## Why these repositories stay separate

The repositories differ along several dimensions that materially change the optimization problem:

1. two-stage versus multistage information structure;
2. scenario probabilities versus uncertainty sets versus distributional ambiguity sets;
3. expectation, service level, worst-case, or other risk criteria;
4. static planning versus sequential policy optimization;
5. exogenous uncertainty versus context-conditioned or learned uncertainty models.

For that reason, sharing the word `stochastic`, `robust`, or `risk` is not a consolidation criterion.
