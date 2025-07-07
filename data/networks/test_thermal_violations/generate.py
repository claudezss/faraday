import pandapower as pp
import pandapower.networks as pn


def create_thermal_violation_network():
    """Create a network with thermal violations - line overloading scenarios."""
    # Start with IEEE 14 bus network for moderate complexity
    net = pn.case14()

    # Add curtailable flag to loads
    net.load["curtailable"] = False
    net.load["name"] = "Load_" + net.load.bus.astype(str)
    net.line["name"] = range(len(net.line["name"]))
    net.line["name"] = net.line["name"].astype(str)

    # Configure some loads as curtailable
    net.load.loc[net.load.bus.isin([4, 5, 9]), "curtailable"] = True

    # Create thermal violations by reducing line capacities and increasing loads
    # Reduce capacity of critical transmission lines
    net.line.loc[net.line.from_bus == 1, "max_i_ka"] = 0.15  # Very low limit
    net.line.loc[net.line.from_bus == 4, "max_i_ka"] = 0.20  # Another bottleneck
    net.line.loc[net.line.to_bus == 5, "max_i_ka"] = 0.18  # Third constraint

    # Increase loads to create violations
    net.load.loc[net.load.bus == 4, "p_mw"] = 0.8  # High load
    net.load.loc[net.load.bus == 5, "p_mw"] = 0.9  # Another high load
    net.load.loc[net.load.bus == 9, "p_mw"] = 0.6  # Third high load

    # Add some switches for reconfiguration options
    pp.create_switch(net, bus=6, element=7, et="b", closed=True, name="SW_6_7")
    pp.create_switch(net, bus=9, element=10, et="b", closed=False, name="SW_9_10")
    pp.create_switch(net, bus=11, element=12, et="b", closed=True, name="SW_11_12")

    return net


def main():
    net = create_thermal_violation_network()
    pp.to_json(net, "net.json")

    # Test that violations exist
    try:
        pp.runpp(net)
        print("✅ Power flow converged")

        # Check for thermal violations
        thermal_violations = 0
        for i, loading in enumerate(net.res_line.loading_percent):
            if loading > 100:
                thermal_violations += 1
                print(f"  Line {i}: {loading:.1f}% loading")

        print(f"Created network with {thermal_violations} thermal violations")

        # Check curtailable loads
        curtailable_count = net.load.curtailable.sum()
        print(f"Available curtailable loads: {curtailable_count}")

        # Check switches
        switch_count = len(net.switch)
        print(f"Available switches: {switch_count}")

    except Exception as e:
        print(f"❌ Power flow failed: {e}")


if __name__ == "__main__":
    main()
