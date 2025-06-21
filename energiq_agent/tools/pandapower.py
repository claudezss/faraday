from typing import Any

import pandapower as pp


def read_network(network_path: str) -> pp.pandapowerNet:
    return pp.from_json(network_path)


def get_load_status(net: pp.pandapowerNet) -> list[dict]:
    loads = [
        {
            "bus": int(row["bus"]),
            "name": row["name"],
            "p_mw": float(row["p_mw"]),
            "q_mvar": float(row["q_mvar"]),
        }
        for _, row in net.load.iterrows()
    ]
    return loads


def get_sync_generator_status(net: pp.pandapowerNet) -> list[dict]:
    sgens = [
        {
            "bus": int(row["bus"]),
            "name": row["name"],
            "p_mw": float(row["p_mw"]),
            "q_mvar": float(row["q_mvar"]),
        }
        for _, row in net.sgen.iterrows()
    ]
    return sgens


def get_switch_status(net: pp.pandapowerNet) -> list[dict]:
    sw = [
        {
            "bus": int(row["bus"]),
            "element": int(row["element"]),
            "name": row["name"],
            "closed": row["closed"],
        }
        for _, row in net.switch.iterrows()
    ]
    return sw


def get_bus_status(net: pp.pandapowerNet) -> list[dict]:
    buses = [
        {
            "index": i,
            "name": row["name"],
            "v_mag_pu": net.res_bus.vm_pu.iloc[i],
            "v_angle_degree": net.res_bus.va_degree.iloc[i],
        }
        for i, row in net.bus.iterrows()
    ]
    return buses


def get_line_status(net: pp.pandapowerNet) -> list[dict]:
    line_loading_percent = [
        {
            "name": row["name"],
            "loading_percent": net.res_line.loading_percent[i],
            "from_bus": row["from_bus"],
            "to_bus": row["to_bus"],
        }
        for i, row in net.line.iterrows()
    ]

    trafo_loading_percent = [
        {
            "name": row["name"],
            "loading_percent": net.res_trafo.loading_percent[i],
            "from_bus": row["hv_bus"],
            "to_bus": row["lv_bus"],
        }
        for i, row in net.trafo.iterrows()
    ]
    return line_loading_percent + trafo_loading_percent


def update_switch_status(network_path, switch_name: str, closed: bool):
    """
    Reconfigure switch status, change switch status to closed or open.

    Args:
        network_path: the path to the network in editing.
        switch_name: the name of the switch to be reconfigured.
        closed: whether to close the switch.
    """
    net = pp.from_json(network_path)
    net.switch.loc[net.switch.name == switch_name, "closed"] = closed
    pp.to_json(net, network_path)

    msg = f"Set switch {switch_name} to {'closed' if closed else 'open'}"

    return {"action": "reconfigure_switch", "detail": msg}


def get_network_status(
    network,
) -> dict[str, Any]:
    """
    Get the status of the network.

    Args:
        network: the pndapower network object..
    """
    net = network
    pp.runpp(net)
    load_status = get_load_status(net)
    sync_generator_status = get_sync_generator_status(net)
    switch_status = get_switch_status(net)
    bus_status = get_bus_status(net)
    line_status = get_line_status(net)

    status = {
        "load_status": load_status,
        "sync_generator_status": sync_generator_status,
        "switch_status": switch_status,
        "bus_status": bus_status,
        "line_status": line_status,
    }

    return status


def curtail_load(network_path: str, load_name: str, curtail_percent: float):
    """
    Curtails the load in an electrical network by reducing its power consumption by a specified percentage.

    This function retrieves an existing network from a file, reduces the power consumption of a
    specific load identified by its name by the given percentage, and then saves the modified network
    back to the file. The percentage reduction is applied multiplicatively to the current value of
    the load's active power.

    Args:
        network_path: the path to the network in editing.
        load_name: the name of the load to be curtailed.
        curtail_percent: the percentage (%) of load power to curtail.
    """
    net = pp.from_json(network_path)
    net.load.loc[net.load.name == load_name, "p_mw"] *= 1 - curtail_percent / 100
    pp.to_json(net, network_path)

    msg = f"Curtail load {load_name} by {curtail_percent}%"

    return {"action": "curtail_load", "detail": msg}


def add_battery(
    network_path: str, bus_index: int, max_energy_kw: float, max_battery_count: int
):
    """
    Add a battery to the network.

    Args:
        network_path: the path to the network in editing.
        bus_index: the index of the bus where the battery is to be added.
        max_energy_kw: the maximum energy capacity of the battery in kW.
        max_battery_count: the maximum number of batteries to be added.
    """
    net = pp.from_json(network_path)
    for i in range(max_battery_count):
        pp.create_storage(
            net,
            bus=bus_index,
            p_mw=-max_energy_kw / 1000,
            max_e_mwh=max_energy_kw / 1000,
        )
    pp.to_json(net, network_path)

    msg = f"Add {max_battery_count} batteries to bus {bus_index} with max energy {max_energy_kw} kW"

    return {"action": "add_battery", "detail": msg}


def run_powerflow(network_path: str):
    """
    Run powerflow on the network.

    Args:
        network_path: the path to the network in editing.

    """
    net = pp.from_json(network_path)
    pp.runpp(net)
    pp.to_json(net, network_path)
    return "success"
