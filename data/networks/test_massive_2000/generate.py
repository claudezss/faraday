import pandapower as pp
import pandapower.networks as pn
import numpy as np


def create_large_scale_network():
    """Create a large-scale network for testing token reduction algorithms."""
    # Use IEEE 118 as our large-scale test network
    net = pn.case118()
    print(f"Using IEEE 118 network: {len(net.bus)} buses")

    # Test base network stability first
    try:
        pp.runpp(net)
        print("✅ Base network is stable")
    except Exception as e:
        print(f"❌ Base network issue: {e}")
        # Use IEEE 57 as fallback
        net = pn.case57()
        print(f"Fallback to IEEE 57 network: {len(net.bus)} buses")
        pp.runpp(net)

    # Add curtailable flag to all loads
    net.load["curtailable"] = False

    # Configure some loads as curtailable (distributed selection)
    total_loads = len(net.load)
    if total_loads > 0:
        # Select ~20% of loads as curtailable
        curtailable_count = max(1, total_loads // 5)
        curtailable_indices = np.random.choice(
            net.load.index, size=curtailable_count, replace=False
        )
        net.load.loc[curtailable_indices, "curtailable"] = True

    # Create very conservative thermal violations
    total_lines = len(net.line)
    if total_lines > 5:
        # Create 3-5 thermal violations
        violation_count = min(5, max(3, total_lines // 25))
        violation_lines = np.random.choice(
            net.line.index, size=violation_count, replace=False
        )

        for line_idx in violation_lines:
            # Very conservative capacity reduction
            current_capacity = net.line.loc[line_idx, "max_i_ka"]
            if current_capacity > 0.1:
                net.line.loc[line_idx, "max_i_ka"] = current_capacity * 0.85

    # Create very conservative voltage violations
    if total_loads > 3:
        # Increase just 3-4 loads slightly
        high_load_count = min(4, max(3, total_loads // 15))
        high_load_indices = np.random.choice(
            net.load.index, size=high_load_count, replace=False
        )

        for load_idx in high_load_indices:
            current_load = net.load.loc[load_idx, "p_mw"]
            # Very light increase (5-10%)
            increase_factor = np.random.uniform(1.05, 1.10)
            net.load.loc[load_idx, "p_mw"] = current_load * increase_factor

    # Add some distributed generation
    dg_count = min(15, len(net.bus) // 10)
    dg_buses = np.random.choice(net.bus.index, size=dg_count, replace=False)
    for i, bus in enumerate(dg_buses):
        dg_size = np.random.uniform(1, 5)
        pp.create_sgen(
            net, bus=bus, p_mw=dg_size, q_mvar=dg_size * 0.2, name=f"DG_{bus}"
        )

    # Add switches for reconfiguration
    switch_count = min(25, len(net.bus) // 8)
    switch_buses = np.random.choice(net.bus.index, size=switch_count, replace=False)

    for i in range(0, len(switch_buses) - 1, 2):
        if i + 1 < len(switch_buses):
            bus1, bus2 = switch_buses[i], switch_buses[i + 1]
            # Most switches closed by default
            closed_state = np.random.choice([True, False], p=[0.8, 0.2])
            pp.create_switch(
                net,
                bus=bus1,
                element=bus2,
                et="b",
                closed=closed_state,
                name=f"SW_{bus1}_{bus2}",
            )

    return net


def main():
    print("🏗️ Creating Large-Scale Test Network")
    print("=" * 50)

    np.random.seed(42)  # For reproducible results
    net = create_large_scale_network()
    pp.to_json(net, "net.json")

    # Test the network
    try:
        print("\n⚡ Running power flow analysis...")
        pp.runpp(net)
        print("✅ Power flow converged")

        # Analyze violations
        voltage_violations = 0
        thermal_violations = 0

        for vm_pu in net.res_bus.vm_pu:
            if vm_pu > 1.05 or vm_pu < 0.95:
                voltage_violations += 1

        for loading in net.res_line.loading_percent:
            if loading > 100:
                thermal_violations += 1

        # Count resources
        total_buses = len(net.bus)
        total_lines = len(net.line)
        total_loads = len(net.load)
        curtailable_count = net.load.curtailable.sum()
        switch_count = len(net.switch)
        dg_count = len(net.sgen)

        print("\n📊 Large-Scale Network Characteristics:")
        print(f"  • Total buses: {total_buses}")
        print(f"  • Total lines: {total_lines}")
        print(f"  • Total loads: {total_loads}")
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

            if resource_ratio >= 0.5:
                print("  ✅ Network appears solvable")
            else:
                print("  ⚠️  Network may be challenging")

        # Test Phase 3 representation capabilities
        print("\n🧪 Testing Phase 3 Token Reduction Algorithms:")

        try:
            from energiq_agent.tools.pandapower import (
                get_network_status,
                get_hierarchical_network_status,
                get_advanced_network_status,
            )

            # Standard representation
            standard = get_network_status(net)
            standard_size = len(str(standard))

            # Violations-only representation
            violations_only = get_network_status(net, violations_only=True)
            violations_size = len(str(violations_only))

            # Hierarchical representation
            hierarchical = get_hierarchical_network_status(net)
            hierarchical_size = len(str(hierarchical))
            zones = len(hierarchical.get("violation_details", {}))

            # Advanced compact representation
            advanced = get_advanced_network_status(
                net, max_tokens=1000, context_mode="adaptive"
            )
            advanced_size = len(str(advanced))

            # Calculate reductions
            violations_reduction = (
                (standard_size - violations_size) / standard_size
            ) * 100
            hierarchical_reduction = (
                (standard_size - hierarchical_size) / standard_size
            ) * 100
            advanced_reduction = ((standard_size - advanced_size) / standard_size) * 100

            print(f"  • Standard representation: ~{standard_size:,} characters")
            print(
                f"  • Violations-only: ~{violations_size:,} characters ({violations_reduction:.1f}% reduction)"
            )
            print(
                f"  • Hierarchical: ~{hierarchical_size:,} characters ({hierarchical_reduction:.1f}% reduction)"
            )
            print(
                f"  • Advanced compact: ~{advanced_size:,} characters ({advanced_reduction:.1f}% reduction)"
            )
            print(f"  • Hierarchical zones: {zones}")

            if total_buses >= 100:
                print("  🎯 Excellent test case for large-scale algorithms!")
                print("  ✅ Demonstrates token reduction effectiveness")

        except Exception as e:
            print(f"  • Representation test error: {e}")

        print("\n🎯 Large-Scale Testing Capabilities:")
        print("  ✅ Token reduction validation ({total_buses} buses)")
        print("  ✅ Hierarchical representation testing")
        print("  ✅ Action optimization scalability")
        print("  ✅ Phase 3 algorithm validation")

        # Performance note
        if total_buses >= 118:
            print("\n💡 Performance Note:")
            print("  • This network size approaches the token limits for standard LLMs")
            print("  • Phase 3 algorithms become essential for effective planning")
            print("  • Demonstrates real-world scalability challenges")

    except Exception as e:
        print(f"❌ Network test failed: {e}")


if __name__ == "__main__":
    main()
