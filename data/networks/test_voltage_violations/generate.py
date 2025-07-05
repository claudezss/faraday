import pandapower as pp
import pandapower.networks as pn


def create_voltage_violation_network():
    """Create a network with voltage violations - under/over voltage scenarios."""
    # Start with a simple radial network for clear voltage profile issues
    net = pn.create_cigre_network_mv()

    # Add curtailable flag to loads
    net.load["curtailable"] = False

    # Configure specific loads as curtailable
    net.load.loc[net.load.bus.isin([7, 9, 11]), "curtailable"] = True

    # Create voltage violations by manipulating load patterns
    # High loads at end of feeders cause voltage drops
    net.load.loc[net.load.bus == 7, "p_mw"] = 4.0  # Very high load
    net.load.loc[net.load.bus == 9, "p_mw"] = 3.5  # Another high load
    net.load.loc[net.load.bus == 11, "p_mw"] = 3.0  # Third high load
    net.load.loc[net.load.bus == 12, "p_mw"] = 2.5  # Fourth load

    # Reduce line capacity to create impedance bottlenecks
    net.line.loc[net.line.to_bus == 7, "r_ohm_per_km"] *= 2.0  # Higher resistance
    net.line.loc[net.line.to_bus == 9, "r_ohm_per_km"] *= 1.8  # Another bottleneck
    net.line.loc[net.line.to_bus == 11, "r_ohm_per_km"] *= 1.5  # Third bottleneck

    # Add some distributed generation that might cause overvoltage
    pp.create_sgen(net, bus=13, p_mw=0.8, q_mvar=0.2, name="PV_13")
    pp.create_sgen(net, bus=14, p_mw=0.6, q_mvar=0.1, name="PV_14")

    # Add switches for network reconfiguration
    pp.create_switch(net, bus=8, element=9, et="b", closed=True, name="SW_8_9")
    pp.create_switch(net, bus=10, element=11, et="b", closed=False, name="SW_10_11")
    pp.create_switch(net, bus=12, element=13, et="b", closed=True, name="SW_12_13")

    return net


def main():
    net = create_voltage_violation_network()
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

        print(f"Created network with {voltage_violations} voltage violations")

        # Check curtailable loads
        curtailable_count = net.load.curtailable.sum()
        print(f"Available curtailable loads: {curtailable_count}")

        # Check switches
        switch_count = len(net.switch)
        print(f"Available switches: {switch_count}")

        # Check DG units
        dg_count = len(net.sgen)
        print(f"Available DG units: {dg_count}")

    except Exception as e:
        print(f"❌ Power flow failed: {e}")


if __name__ == "__main__":
    main()
