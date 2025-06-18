from pathlib import Path

import pandapower as pp

from mcp.server.fastmcp import FastMCP
import json

mcp = FastMCP("pandapower_api")

NETWORK_DIR = Path(__file__).parent.parent.parent / "data" / "networks" / "cigre_mv"

TEMP_NETWORK_DIR = Path(__file__).parent.parent.parent / "data" / "networks" / "temp"

TEMP_NETWORK_DIR.mkdir(parents=True, exist_ok=True)

NETWORK_PATH: dict[str, Path] = {"modified_cigre_mv": NETWORK_DIR / "net.json"}


@mcp.resource("resource://networks")
def get_available_networks() -> list[str]:
    """
    Get list of pandapower network name
    """
    return list(NETWORK_PATH.keys())


def get_net(network_name: str) -> pp.pandapowerNet:
    return pp.from_json(str(NETWORK_PATH[network_name].absolute()))


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


@mcp.tool()
def get_path_to_network_in_editing() -> str:
    """
    Get network path in editing
    """
    from uuid import uuid4

    net = get_net("modified_cigre_mv")
    uuid_str = str(uuid4())
    net_path = TEMP_NETWORK_DIR / uuid_str
    net_path.mkdir(parents=True, exist_ok=True)
    name = net_path / "net.json"
    pp.to_json(net, name)
    record_path = name.parent / "record.json"
    with open(record_path, "w") as f:
        json.dump([], f)
    return str(name.absolute())


def write_record(network_path: str, record: dict):
    record_path = Path(network_path).parent / "record.json"
    action_record = json.loads(open(record_path).read())
    action_record.append(record)
    with open(record_path, "+w") as f:
        json.dump(action_record, f)


@mcp.tool()
def update_switch_status(network_path: str, switch_name: str, closed: bool):
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
    write_record(network_path, {"action": "reconfigure_switch", "detail": msg})
    return "success"


@mcp.tool()
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
    write_record(network_path, {"action": "curtail_load", "detail": msg})
    return "success"


@mcp.tool()
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
    write_record(network_path, {"action": "add_battery", "detail": msg})
    return "success"


@mcp.tool()
def run_powerflow_on_edited_network(network_path: str):
    """
    Run powerflow on the edited network.

    Args:
        network_path: the path to the network in editing.

    """
    net = pp.from_json(network_path)
    pp.runpp(net)
    pp.to_json(net, network_path)

    return "success"


@mcp.tool()
def get_edited_network_status(network_path: str) -> str:
    """
    Save the status of the edited network.

    Args:
        network_path: the path to the network in editing.
    """
    net = pp.from_json(network_path)
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
    import json

    path = Path(network_path).parent / "status.json"
    with open(path, "w") as f:
        json.dump(status, f)
    return f"Status was saved to {path.absolute()}"


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
