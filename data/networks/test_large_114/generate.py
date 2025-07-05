import pandapower as pp
import pandapower.networks as pn
import numpy as np


def create_large_114_network():
    """Create a 114-node network with strategic violations for testing medium-scale optimization."""
    # Use IEEE 39 bus network as base and replicate/extend it
    net = pn.case39()

    # Check if we need to extend to reach 114 buses
    current_buses = len(net.bus)
    print(f"Base network has {current_buses} buses")

    # If we need more buses, create additional feeders
    if current_buses < 100:
        # Add radial feeders to existing buses to reach ~114 buses
        base_buses = list(net.bus.index)

        # Add feeders from strategic buses
        feeder_roots = base_buses[::5]  # Every 5th bus gets a feeder

        for root_bus in feeder_roots[:10]:  # Limit to 10 feeders
            # Create 5-8 buses per feeder
            feeder_length = np.random.randint(5, 9)

            for i in range(feeder_length):
                new_bus_idx = len(net.bus)
                pp.create_bus(net, vn_kv=138, name=f"Feeder_{root_bus}_Bus_{i}")

                # Connect to previous bus in feeder
                from_bus = root_bus if i == 0 else (new_bus_idx - 1)
                pp.create_line_from_parameters(
                    net,
                    from_bus=from_bus,
                    to_bus=new_bus_idx,
                    length_km=np.random.uniform(5, 20),
                    r_ohm_per_km=0.1,
                    x_ohm_per_km=0.4,
                    c_nf_per_km=10,
                    max_i_ka=0.5,
                    name=f"Feeder_{root_bus}_Line_{i}",
                )

                # Add load to some buses
                if i > 0 and np.random.random() > 0.3:  # 70% chance of load
                    load_size = np.random.uniform(2, 10)
                    pp.create_load(
                        net,
                        bus=new_bus_idx,
                        p_mw=load_size,
                        q_mvar=load_size * 0.3,
                        name=f"Load_F{root_bus}_B{i}",
                    )

    # Now configure violations and resources
    net.load["curtailable"] = False

    # Configure strategic loads as curtailable
    total_loads = len(net.load)
    curtailable_count = min(12, total_loads // 8)  # ~12 curtailable loads
    curtailable_indices = np.random.choice(
        net.load.index, size=curtailable_count, replace=False
    )
    net.load.loc[curtailable_indices, "curtailable"] = True

    # Create thermal violations by reducing line capacities moderately
    total_lines = len(net.line)
    violation_lines = min(5, total_lines // 20)  # ~5 thermal violations
    critical_lines = np.random.choice(
        net.line.index, size=violation_lines, replace=False
    )

    for line_idx in critical_lines:
        # Reduce capacity moderately
        if "max_i_ka" in net.line.columns:
            current_capacity = net.line.loc[line_idx, "max_i_ka"]
            if current_capacity > 0.1:
                net.line.loc[line_idx, "max_i_ka"] = (
                    current_capacity * 0.6
                )  # Less aggressive
        else:
            net.line.loc[line_idx, "max_i_ka"] = 0.3

    # Create voltage violations by increasing some loads moderately
    high_load_indices = np.random.choice(
        net.load.index, size=min(6, len(net.load)), replace=False
    )
    for load_idx in high_load_indices:
        current_load = net.load.loc[load_idx, "p_mw"]
        # Increase load by 30-60% (more conservative)
        increase_factor = np.random.uniform(1.3, 1.6)
        net.load.loc[load_idx, "p_mw"] = current_load * increase_factor

    # Add distributed generation
    dg_buses = np.random.choice(
        net.bus.index, size=min(15, len(net.bus) // 8), replace=False
    )
    for i, bus in enumerate(dg_buses):
        dg_size = np.random.uniform(3, 12)
        pp.create_sgen(
            net, bus=bus, p_mw=dg_size, q_mvar=dg_size * 0.2, name=f"DG_{bus}"
        )

    # Add switches for reconfiguration
    switch_buses = np.random.choice(
        net.bus.index, size=min(20, len(net.bus) // 6), replace=False
    )
    for i, bus in enumerate(switch_buses):
        if i < len(switch_buses) - 1:
            target_bus = switch_buses[i + 1]
            closed_state = np.random.choice([True, False])
            pp.create_switch(
                net,
                bus=bus,
                element=target_bus,
                et="b",
                closed=closed_state,
                name=f"SW_{bus}_{target_bus}",
            )

    return net


def main():
    np.random.seed(42)  # For reproducible results
    net = create_large_114_network()
    pp.to_json(net, "net.json")

    # Test that the network is solvable
    try:
        pp.runpp(net)
        print("✅ Power flow converged")

        # Check for violations
        voltage_violations = 0
        thermal_violations = 0

        for i, (bus_idx, vm_pu) in enumerate(zip(net.bus.index, net.res_bus.vm_pu)):
            if vm_pu > 1.05 or vm_pu < 0.95:
                voltage_violations += 1

        for i, loading in enumerate(net.res_line.loading_percent):
            if loading > 100:
                thermal_violations += 1

        if len(net.trafo) > 0:
            for i, loading in enumerate(net.res_trafo.loading_percent):
                if loading > 100:
                    thermal_violations += 1

        # Check available resources
        curtailable_count = net.load.curtailable.sum()
        switch_count = len(net.switch)
        dg_count = len(net.sgen)
        total_buses = len(net.bus)
        total_lines = len(net.line)

        print("📊 Large Network (~114-node) Characteristics:")
        print(f"  • Total buses: {total_buses}")
        print(f"  • Total lines: {total_lines}")
        print(f"  • Voltage violations: {voltage_violations}")
        print(f"  • Thermal violations: {thermal_violations}")
        print(f"  • Total violations: {voltage_violations + thermal_violations}")
        print(f"  • Curtailable loads: {curtailable_count}")
        print(f"  • Switches: {switch_count}")
        print(f"  • DG units: {dg_count}")

        # Solvability assessment
        total_violations = voltage_violations + thermal_violations
        total_resources = curtailable_count + switch_count + 3  # +3 for max batteries

        if total_violations > 0:
            resource_ratio = total_resources / total_violations
            print(f"  • Resource ratio: {resource_ratio:.1f}")

            if resource_ratio >= 0.3:  # Lower threshold for large networks
                print("  ✅ Network appears solvable (medium-scale optimization test)")
            else:
                print("  ⚠️  Network may require advanced optimization strategies")

        # Test representation modes
        print("\n🧪 Testing Phase 3 representation modes:")

        try:
            # Test hierarchical representation
            from energiq_agent.tools.pandapower import get_hierarchical_network_status

            hierarchical = get_hierarchical_network_status(net)
            print(
                f"  • Hierarchical zones: {len(hierarchical.get('violation_details', {}))}"
            )

            # Test advanced representation
            from energiq_agent.tools.pandapower import get_advanced_network_status

            _ = get_advanced_network_status(net, max_tokens=1500, context_mode="graph")
            print("  • Graph representation available: ✅")
            print("  • Token reduction features: ✅")

        except Exception as e:
            print(f"  • Representation test: ⚠️ {e}")

    except Exception as e:
        print(f"❌ Network test failed: {e}")


if __name__ == "__main__":
    main()
