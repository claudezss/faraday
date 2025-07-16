import pandapower as pp
import pandapower.networks as pn


def create_mixed_violation_network():
    """Create a network with both voltage and thermal violations - complex optimization scenario."""
    # Start with CIGRE MV network and extend it for mixed violations
    net = pn.create_cigre_network_mv(with_der="pv_wind")

    # Add curtailable flag to loads
    net.load["curtailable"] = False

    # Configure strategic loads as curtailable (distributed across network)
    net.load.loc[net.load.bus.isin([7, 9, 11]), "curtailable"] = True

    # Create thermal violations by reducing line capacities
    net.line.loc[net.line.from_bus == 2, "max_i_ka"] = 0.15  # Critical line
    net.line.loc[net.line.from_bus == 6, "max_i_ka"] = 0.18  # Another bottleneck
    net.line.loc[net.line.to_bus == 9, "max_i_ka"] = 0.12  # Third constraint

    # Create voltage violations by increasing loads moderately
    net.load.loc[net.load.bus == 7, "p_mw"] = 2.5  # High load for voltage drop
    net.load.loc[net.load.bus == 9, "p_mw"] = 2.2  # Another high load
    net.load.loc[net.load.bus == 11, "p_mw"] = 2.0  # Third high load
    net.load.loc[net.load.bus == 12, "p_mw"] = 1.8  # Fourth high load

    # Add some distributed generation at end buses
    pp.create_sgen(net, bus=13, p_mw=1.2, q_mvar=0.2, name="PV_13")
    pp.create_sgen(net, bus=14, p_mw=0.8, q_mvar=0.1, name="Wind_14")

    # Add switches for complex reconfiguration scenarios
    pp.create_switch(net, bus=8, element=9, et="b", closed=True, name="SW_8_9")
    pp.create_switch(net, bus=10, element=11, et="b", closed=False, name="SW_10_11")
    pp.create_switch(net, bus=12, element=13, et="b", closed=True, name="SW_12_13")
    pp.create_switch(net, bus=6, element=7, et="b", closed=False, name="SW_6_7")

    # Modify transformer settings to create additional complexity
    if len(net.trafo) > 0:
        net.trafo.loc[0, "tap_pos"] = -2  # Adjust tap position

    return net


def main():
    net = create_mixed_violation_network()
    pp.to_json(net, "net.json")

    # Test that violations exist
    try:
        pp.runpp(net)
        print("✅ Power flow converged")

        # Check for voltage violations
        voltage_violations = 0
        for i, (bus_idx, vm_pu) in enumerate(zip(net.bus.index, net.res_bus.vm_pu)):
            if vm_pu > 1.05 or vm_pu < 0.95:
                voltage_violations += 1
                violation_type = "overvoltage" if vm_pu > 1.05 else "undervoltage"
                print(f"  Bus {bus_idx}: {vm_pu:.3f} pu ({violation_type})")

        # Check for thermal violations
        thermal_violations = 0
        for i, loading in enumerate(net.res_line.loading_percent):
            if loading > 100:
                thermal_violations += 1
                print(f"  Line {i}: {loading:.1f}% loading")

        # Check transformers
        if len(net.trafo) > 0:
            for i, loading in enumerate(net.res_trafo.loading_percent):
                if loading > 100:
                    thermal_violations += 1
                    print(f"  Transformer {i}: {loading:.1f}% loading")

        print(
            f"Created network with {voltage_violations} voltage violations and {thermal_violations} thermal violations"
        )

        # Check resources
        curtailable_count = net.load.curtailable.sum()
        print(f"Available curtailable loads: {curtailable_count}")

        switch_count = len(net.switch)
        print(f"Available switches: {switch_count}")

        dg_count = len(net.sgen)
        print(f"Available DG units: {dg_count}")

        total_buses = len(net.bus)
        print(f"Total buses: {total_buses}")

    except Exception as e:
        print(f"❌ Power flow failed: {e}")


if __name__ == "__main__":
    main()
