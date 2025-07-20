from pathlib import Path

import pandas as pd
import pandapower as pp


def create_ieee69_network(csv_file_path="ieee69.csv"):
    """
    Create IEEE 69-bus test system using pandapower.

    Parameters:
    csv_file_path (str): Path to the CSV file containing network data

    Returns:
    pandapower.pandapowerNet: The created network
    """

    # Read the CSV data
    df = pd.read_csv(csv_file_path)

    # Create empty pandapower network
    net = pp.create_empty_network(name="IEEE 69-Bus Test System")

    # IEEE 69-bus system parameters
    base_voltage = 12.66  # kV

    # Create buses (69 buses total: 1 to 69)
    # Bus 1 is the slack bus (substation)
    for i in range(1, 70):  # Buses 1 to 69
        if i == 1:
            # Slack bus (substation)
            pp.create_bus(net, vn_kv=base_voltage, name=f"Bus_{i}", type="b")
        else:
            # Load buses
            pp.create_bus(net, vn_kv=base_voltage, name=f"Bus_{i}", type="b")

    # Create external grid connection at bus 1 (slack bus)
    pp.create_ext_grid(net, bus=0, vm_pu=1.03, va_degree=0.0, name="External Grid")

    # Create lines from CSV data
    for idx, row in df.iterrows():
        from_bus = int(row["from"]) - 1  # Convert to 0-based indexing
        to_bus = int(row["to"]) - 1  # Convert to 0-based indexing
        r_ohm_per_km = row["rohm"]  # Resistance in Ohm
        x_ohm_per_km = row["xohm"]  # Reactance in Ohm
        max_i_ka = row["maxi"] / 1000  # Convert from A to kA

        # Create line with 1 km length (since impedances are already in absolute values)
        pp.create_line_from_parameters(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=1.0,  # 1 km since impedances are absolute
            r_ohm_per_km=r_ohm_per_km,
            x_ohm_per_km=x_ohm_per_km,
            c_nf_per_km=0,  # No capacitance data provided
            max_i_ka=max_i_ka,
            name=f"Line_{row['from']}-{row['to']}",
        )

    # Create loads from CSV data (P and Q values)
    for idx, row in df.iterrows():
        to_bus = int(row["to"]) - 1  # Convert to 0-based indexing
        p_mw = row["P"] / 1000  # Convert from kW to MW
        q_mvar = row["Q"] / 1000  # Convert from kVar to MVar

        # Only create load if P or Q is non-zero
        if p_mw != 0 or q_mvar != 0:
            pp.create_load(
                net, bus=to_bus, p_mw=p_mw, q_mvar=q_mvar, name=f"Load_Bus_{row['to']}"
            )
    sw_map = [(50, 59), (11, 44), (13, 21), (65, 27)]

    for f, t in sw_map:
        line_idx = pp.create_line_from_parameters(
            net,
            from_bus=f - 1,
            to_bus=t - 1,
            length_km=1.0,
            r_ohm_per_km=0.12,
            x_ohm_per_km=0.06,
            c_nf_per_km=0,
            max_i_ka=0.2,  # Lower rating for tie lines
            name=f"Tie_Line_{f}-{t}",
        )
        pp.create_switch(
            net,
            bus=f - 1,
            element=line_idx,
            et="l",
            closed=False,
            name=f"Tie_Line_Switch_{f}-{t}",
        )

    return net


def print_network_summary(net):
    """Print a summary of the created network."""
    print("=" * 60)
    print("IEEE 69-Bus Test System - Network Summary")
    print("=" * 60)
    print(f"Number of buses: {len(net.bus)}")
    print(f"Number of lines: {len(net.line)}")
    print(f"Number of loads: {len(net.load)}")
    print(f"Number of witches: {len(net.switch)}")
    print(f"Number of external grids: {len(net.ext_grid)}")


# Example usage
if __name__ == "__main__":
    # Create the network
    print("Creating IEEE 69-bus network...")
    net = create_ieee69_network(str(Path(__file__).parent / "ieee69.csv"))
    pp.runpp(net)

    pp.to_json(net, Path(__file__).parent / "ieee69.json")

    # Print network summary
    print_network_summary(net)

    pp.plotting.plotly.pf_res_plotly(net, auto_open=True)
