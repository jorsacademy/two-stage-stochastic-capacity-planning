from src.model import sample_data, stochastic_metrics


def main() -> None:
    data = sample_data()
    metrics = stochastic_metrics(data)
    rp = metrics.recourse_problem

    print("Two-Stage Stochastic Capacity Planning")
    print("=" * 39)
    print("\nOptimal first-stage capacities:")
    for name, units in rp.capacities.items():
        print(f"  {name}: {units} unit(s)")

    print(f"\nRecourse problem (RP):      {rp.objective:,.2f}")
    print(f"Expected-value evaluation: {metrics.eev_objective:,.2f}")
    print(f"Wait-and-see (WS):         {metrics.wait_and_see_objective:,.2f}")
    print(f"Value of stochastic solution (VSS): {metrics.vss:,.2f}")
    print(f"Expected value of perfect information (EVPI): {metrics.evpi:,.2f}")

    print("\nScenario recourse decisions:")
    for scenario in data.scenarios:
        name = scenario.name
        internal_total = sum(
            rp.internal[cap.name, name] for cap in data.capacity_types
        )
        print(
            f"  {name:>6}: internal={internal_total:6.1f}, "
            f"outsourced={rp.outsourced[name]:5.1f}, "
            f"unmet={rp.unmet[name]:5.1f}"
        )


if __name__ == "__main__":
    main()
