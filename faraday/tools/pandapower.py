from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandapower as pp
import json

from collections import defaultdict
from enum import Enum
from dataclasses import dataclass
from faraday.agents.workflow.state import NetworkState, Violation


@dataclass
class VoltageThresholds:
    """Configuration class for voltage violation thresholds."""

    v_max: float = 1.05  # Maximum allowed voltage in per unit
    v_min: float = 0.95  # Minimum allowed voltage in per unit

    # Severity thresholds
    critical_high: float = 1.15
    critical_low: float = 0.85
    high_violation_upper: float = 1.1
    high_violation_lower: float = 0.9
    medium_violation_upper: float = 1.05
    medium_violation_lower: float = 0.95


# Global default voltage thresholds - can be overridden
_DEFAULT_VOLTAGE_THRESHOLDS = VoltageThresholds()


def set_voltage_thresholds(v_max: float = 1.05, v_min: float = 0.95) -> None:
    """Set global voltage violation thresholds.

    Args:
        v_max: Maximum allowed voltage in per unit (default: 1.05)
        v_min: Minimum allowed voltage in per unit (default: 0.95)
    """
    global _DEFAULT_VOLTAGE_THRESHOLDS
    _DEFAULT_VOLTAGE_THRESHOLDS.v_max = v_max
    _DEFAULT_VOLTAGE_THRESHOLDS.v_min = v_min
    # Auto-adjust severity thresholds based on new limits
    _DEFAULT_VOLTAGE_THRESHOLDS.medium_violation_upper = v_max
    _DEFAULT_VOLTAGE_THRESHOLDS.medium_violation_lower = v_min


def get_voltage_thresholds() -> VoltageThresholds:
    """Get current voltage violation thresholds."""
    return _DEFAULT_VOLTAGE_THRESHOLDS


def load_action_log(network_path: str) -> list[dict]:
    log_path = Path(network_path).parent / "action_log.json"
    if log_path.exists():
        with open(log_path, "r") as f:
            logs = json.load(f)
    else:
        logs = []
    return logs


def record_action(network_path: str, log: dict) -> None:
    logs = load_action_log(network_path)
    log_path = Path(network_path).parent / "action_log.json"
    logs += [log]
    with open(log_path, "+w") as f:
        json.dump(logs, f)
    return None


def rest_action_log(network_path: str) -> None:
    log_path = Path(network_path).parent / "action_log.json"
    with open(log_path, "+w") as f:
        json.dump([], f)
    return None


def read_network(network_path: str) -> pp.pandapowerNet:
    return pp.from_json(network_path)


def get_load_status(net: pp.pandapowerNet) -> list[dict]:
    loads = [
        {
            "bus_idx": int(row["bus"]),
            "name": row["name"],
            "p_mw": round(float(row["p_mw"]), 3),
            "q_mvar": round(float(row["q_mvar"]), 3),
            "curtailable": bool(row.get("curtailable", False)),
        }
        for _, row in net.load.iterrows()
    ]
    return loads


def get_sync_generator_status(net: pp.pandapowerNet) -> list[dict]:
    sgens = [
        {
            "bus_idx": int(row["bus"]),
            "name": row["name"],
            "p_mw": round(float(row["p_mw"]), 3),
            "q_mvar": round(float(row["q_mvar"]), 3),
            "controllable": bool(row.get("controllable", False)),
        }
        for _, row in net.sgen.iterrows()
    ]
    return sgens


def get_switch_status(net: pp.pandapowerNet) -> list[dict]:
    sw = [
        {
            "from_bus_idx": int(row["bus"]),
            "to_bus_idx": int(row["element"]),
            "name": row["name"],
            "closed": row["closed"],
            "controllable": bool(row.get("controllable", True)),
        }
        for _, row in net.switch[net.switch.et == "l"].iterrows()
    ]
    return sw


def get_bus_status(net: pp.pandapowerNet) -> list[dict]:
    buses = [
        {
            "idx": i,
            "name": row["name"],
            "v_mag_pu": round(float(net.res_bus.vm_pu.iloc[idx]), 3),
        }
        for idx, (i, row) in enumerate(net.bus.iterrows())
    ]
    return buses


def get_line_status(net: pp.pandapowerNet) -> list[dict]:
    line_loading_percent = [
        {
            "name": row["name"],
            "loading_percent": round(float(net.res_line.loading_percent.iloc[idx]), 3),
            "from_bus_idx": int(row["from_bus"]),
            "to_bus_idx": int(row["to_bus"]),
        }
        for idx, (i, row) in enumerate(net.line.iterrows())
    ]

    trafo_loading_percent = [
        {
            "name": row["name"],
            "loading_percent": round(float(net.res_trafo.loading_percent.iloc[idx]), 3),
            "from_bus_idx": int(row["hv_bus"]),
            "to_bus_idx": int(row["lv_bus"]),
        }
        for idx, (i, row) in enumerate(net.trafo.iterrows())
    ]
    return line_loading_percent + trafo_loading_percent


def get_network_status(
    network,
) -> NetworkState:
    """
    Get the status of the network.

    Args:
        network: the pandapower network object.
    """
    net = network
    pp.runpp(net)

    load_status = get_load_status(net)
    sync_generator_status = get_sync_generator_status(net)
    switch_status = get_switch_status(net)
    bus_status = get_bus_status(net)
    line_status = get_line_status(net)

    status = defaultdict(
        buses=bus_status,
        switches=switch_status,
        lines=line_status,
        loads=load_status,
        generators=sync_generator_status,
    )

    return NetworkState.model_validate(status)


def get_json_network_status(net):
    pp.runpp(net)

    data = {
        "buses": [],
        "lines": [],
        "transformers": [],
        "switches": [],
        "generators": [],
        "loads": [],
        "violations": {"voltage_violations": [], "thermal_violations": []},
    }

    thresholds = get_voltage_thresholds()

    # Buses with voltages
    for idx, row in net.bus.iterrows():
        vm_pu = net.res_bus.loc[idx, "vm_pu"]
        bus_data = {
            "id": idx,
            "name": row["name"],
            "voltage_pu": round(vm_pu, 4),
            "voltage_violation": str(
                vm_pu < thresholds.v_min or vm_pu > thresholds.v_max
            ),
        }
        data["buses"].append(bus_data)

    # Lines with loading %
    for idx, row in net.line.iterrows():
        loading = net.res_line.loc[idx, "loading_percent"]
        line_data = {
            "id": idx,
            "name": row["name"],
            "from_bus": row.from_bus,
            "to_bus": row.to_bus,
            "loading_percent": round(loading, 2),
            "thermal_violation": str(loading > 100),
        }
        data["lines"].append(line_data)

    # Transformers
    for idx, row in net.trafo.iterrows():
        loading = net.res_trafo.loc[idx, "loading_percent"]
        trafo_data = {
            "id": idx,
            "name": row["name"],
            "hv_bus": row.hv_bus,
            "lv_bus": row.lv_bus,
            "tap_pos": row.tap_pos,
            "loading_percent": round(loading, 2),
            "thermal_violation": str(loading > 100.0),
        }

        data["transformers"].append(trafo_data)

    # Switches
    for idx, row in net.switch.iterrows():
        switch = {
            "id": idx,
            "name": row["name"],
            "bus": row.bus,
            "element": row.element,
            "status": "closed" if row.closed else "open",
        }
        data["switches"].append(switch)

    # Loads
    for idx, row in net.load.iterrows():
        load = {
            "id": idx,
            "name": row["name"],
            "bus": row.bus,
            "p_mw": row.p_mw,
            "q_mvar": row.q_mvar,
            "status": "connected" if row.in_service else "disconnected",
            "curtailable": row.get("curtailable", False),
        }
        data["loads"].append(load)

    # Generators
    for idx, row in net.sgen.iterrows():
        gen = {
            "id": idx,
            "name": row["name"],
            "bus": row.bus,
            "p_mw": row.p_mw,
            "q_mvar": row.q_mvar,
            "type": row.get("type", "sgen"),
            "status": "connected" if row.in_service else "disconnected",
        }
        data["generators"].append(gen)

    voltage_violations = []
    thermal_violations = []

    for idx, (i, row) in enumerate(net.bus.iterrows()):
        v_mag = net.res_bus.vm_pu.iloc[idx]
        if v_mag > thresholds.v_max or v_mag < thresholds.v_min:
            voltage_violations.append(
                {
                    "bus": i,
                    "name": row["name"],
                    "v_mag_pu": round(float(v_mag), 3),
                    "severity": "high"
                    if v_mag > thresholds.high_violation_upper
                    or v_mag < thresholds.high_violation_lower
                    else "medium",
                }
            )

    for idx, (i, row) in enumerate(net.line.iterrows()):
        loading = net.res_line.loading_percent.iloc[idx]
        if loading > 100:
            thermal_violations.append(
                {
                    "name": row["name"],
                    "loading_percent": round(float(loading), 3),
                    "from_bus": int(row["from_bus"]),
                    "to_bus": int(row["to_bus"]),
                    "severity": "high" if loading > 120 else "medium",
                }
            )

    for idx, (i, row) in enumerate(net.trafo.iterrows()):
        loading = net.res_trafo.loading_percent.iloc[idx]
        if loading > 100:
            thermal_violations.append(
                {
                    "name": row["name"],
                    "loading_percent": round(float(loading), 3),
                    "from_bus": int(row["hv_bus"]),
                    "to_bus": int(row["lv_bus"]),
                    "severity": "high" if loading > 120 else "medium",
                }
            )
    data["violations"] = {}
    data["violations"]["voltage"] = voltage_violations
    data["violations"]["thermal"] = thermal_violations
    return data


def get_violations(
    network: pp.pandapowerNet,
    v_max: float = 1.05,
    v_min: float = 0.95,
    thermal_limit: float = 100,
) -> Violation:
    pp.runpp(network)

    v_viola = network.res_bus[
        (network.res_bus.vm_pu > v_max) | (network.res_bus.vm_pu < v_min)
    ]

    v_viola_dicts = [
        {
            "bus_idx": i,
            "v_mag_pu": v["vm_pu"],
        }
        for i, v in v_viola.iterrows()
    ]

    network.res_line["name"] = network.line.name
    network.res_line["from_bus"] = network.line.from_bus
    network.res_line["to_bus"] = network.line.to_bus

    network.res_trafo["name"] = network.trafo.name
    network.res_trafo["from_bus"] = network.trafo.hv_bus
    network.res_trafo["to_bus"] = network.trafo.lv_bus

    line_thermal_viola = network.res_line[
        network.res_line.loading_percent > thermal_limit
    ]
    trafo_thermal_viola = network.res_trafo[
        network.res_trafo.loading_percent > thermal_limit
    ]

    thermal_viola_dicts = [
        {
            "name": v["name"],
            "from_bus_idx": v["from_bus"],
            "to_bus_idx": v["to_bus"],
            "loading_percent": v["loading_percent"],
        }
        for i, v in chain(line_thermal_viola.iterrows(), trafo_thermal_viola.iterrows())
    ]

    disconnected_buses = network.bus.index[network.res_bus.vm_pu.isna()].tolist()

    return Violation.model_validate(
        {
            "voltage": v_viola_dicts,
            "thermal": thermal_viola_dicts,
            "disconnected_buses": disconnected_buses,
        }
    )


def get_electrical_zones(
    net: pp.pandapowerNet, zone_size: int = 50
) -> dict[int, list[int]]:
    """
    Partition network into electrical zones based on voltage levels and connectivity.

    Args:
        net: pandapower network object
        zone_size: target size for each zone

    Returns:
        Dictionary mapping zone_id to list of bus indices
    """
    zones = defaultdict(list)

    # Simple voltage-level based partitioning
    voltage_levels = net.bus.vn_kv.unique()

    zone_id = 0
    for vl in voltage_levels:
        buses_at_level = net.bus[net.bus.vn_kv == vl].index.tolist()

        # Further partition large voltage levels
        for i in range(0, len(buses_at_level), zone_size):
            zone_buses = buses_at_level[i : i + zone_size]
            zones[zone_id] = zone_buses
            zone_id += 1

    return dict(zones)


def get_electrical_distance(net: pp.pandapowerNet, from_bus: int, to_bus: int) -> int:
    """
    Calculate electrical distance between two buses (number of branches).

    Args:
        net: pandapower network object
        from_bus: source bus index
        to_bus: target bus index

    Returns:
        Electrical distance (number of hops)
    """
    # Build adjacency graph
    adjacency = defaultdict(set)

    # Add lines
    for _, line in net.line.iterrows():
        adjacency[line.from_bus].add(line.to_bus)
        adjacency[line.to_bus].add(line.from_bus)

    # Add transformers
    for _, trafo in net.trafo.iterrows():
        adjacency[trafo.hv_bus].add(trafo.lv_bus)
        adjacency[trafo.lv_bus].add(trafo.hv_bus)

    # BFS to find shortest path
    if from_bus == to_bus:
        return 0

    queue = [(from_bus, 0)]
    visited = {from_bus}

    while queue:
        current_bus, distance = queue.pop(0)

        for neighbor in adjacency[current_bus]:
            if neighbor == to_bus:
                return distance + 1

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

    return float("inf")  # No path found


def get_hierarchical_network_status(net: pp.pandapowerNet) -> dict[str, Any]:
    """
    Get hierarchical network status for large networks.

    Args:
        net: pandapower network object

    Returns:
        Hierarchical network representation
    """
    # Network overview
    total_buses = len(net.bus)
    total_lines = len(net.line) + len(net.trafo)
    total_loads = len(net.load)
    total_generators = len(net.sgen)

    # Identify violations
    voltage_violations = []
    thermal_violations = []

    thresholds = get_voltage_thresholds()
    for idx, (i, row) in enumerate(net.bus.iterrows()):
        v_mag = net.res_bus.vm_pu.iloc[idx]
        if v_mag > thresholds.v_max or v_mag < thresholds.v_min:
            voltage_violations.append(
                {
                    "bus": i,
                    "name": row["name"],
                    "v_mag_pu": round(float(v_mag), 3),
                    "severity": "high"
                    if v_mag > thresholds.high_violation_upper
                    or v_mag < thresholds.high_violation_lower
                    else "medium",
                }
            )

    for idx, (i, row) in enumerate(net.line.iterrows()):
        loading = net.res_line.loading_percent.iloc[idx]
        if loading > 100:
            thermal_violations.append(
                {
                    "name": row["name"],
                    "loading_percent": round(float(loading), 3),
                    "from_bus": int(row["from_bus"]),
                    "to_bus": int(row["to_bus"]),
                    "severity": "high" if loading > 120 else "medium",
                }
            )

    for idx, (i, row) in enumerate(net.trafo.iterrows()):
        loading = net.res_trafo.loading_percent.iloc[idx]
        if loading > 100:
            thermal_violations.append(
                {
                    "name": row["name"],
                    "loading_percent": round(float(loading), 3),
                    "from_bus": int(row["hv_bus"]),
                    "to_bus": int(row["lv_bus"]),
                    "severity": "high" if loading > 120 else "medium",
                }
            )

    # Get zones with violations
    zones = get_electrical_zones(net)
    violation_zones = {}

    for zone_id, bus_list in zones.items():
        zone_violations = [v for v in voltage_violations if v["bus"] in bus_list]
        zone_thermal = [
            t
            for t in thermal_violations
            if t["from_bus"] in bus_list or t["to_bus"] in bus_list
        ]

        if zone_violations or zone_thermal:
            # Find controllable resources in this zone
            controllable_loads = []
            for _, load in net.load.iterrows():
                if load["bus"] in bus_list and load.get("curtailable", False):
                    controllable_loads.append(
                        {
                            "name": load["name"],
                            "bus": int(load["bus"]),
                            "p_mw": round(float(load["p_mw"]), 3),
                        }
                    )

            zone_switches = []
            for _, switch in net.switch.iterrows():
                if switch["bus"] in bus_list:
                    zone_switches.append(
                        {
                            "name": switch["name"],
                            "bus": int(switch["bus"]),
                            "element": int(switch["element"]),
                            "closed": bool(switch["closed"]),
                        }
                    )

            violation_zones[f"zone_{zone_id}"] = {
                "buses": bus_list,
                "voltage_violations": zone_violations,
                "thermal_violations": zone_thermal,
                "controllable_loads": controllable_loads,
                "switches": zone_switches,
                "total_buses": len(bus_list),
            }

    return {
        "network_overview": {
            "total_buses": total_buses,
            "total_lines": total_lines,
            "total_loads": total_loads,
            "total_generators": total_generators,
            "total_voltage_violations": len(voltage_violations),
            "total_thermal_violations": len(thermal_violations),
            "violation_zones": list(violation_zones.keys()),
            "healthy_zones": [
                f"zone_{zid}"
                for zid in zones.keys()
                if f"zone_{zid}" not in violation_zones
            ],
        },
        "violation_details": violation_zones,
    }


def update_switch_status(network_path: str, switch_name: str, closed: bool):
    """
    Reconfigure switch status, change switch status to closed or open.

    Args:
        network_path: the path to the network in editing.
        switch_name: the name of the switch to be reconfigured.
        closed: whether to close the switch.
    """
    msg = f"Set switch {switch_name} to {'closed' if closed else 'open'}"
    log = {"action": "reconfigure_switch", "detail": msg}

    logs = load_action_log(network_path)
    if log in logs:
        return {"action": "None", "detail": "This action has been executed before."}

    net = pp.from_json(network_path)
    net.switch.loc[net.switch.name == switch_name, "closed"] = closed

    pp.to_json(net, network_path)
    record_action(network_path, log)
    return log


def curtail_load(
    network_path: str,
    load_name: str,
    curtail_percent: float,
    bus_index: int = None,
):
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
        bus_index: the bus index of the load to be curtailed. If load_name is provided, this argument is ignored.
    """
    msg = f"Curtail load {load_name} by {curtail_percent}%"
    log = {"action": "curtail_load", "detail": msg}

    logs = load_action_log(network_path)

    if log in logs:
        return {"action": "None", "detail": "This action has been executed before."}

    net = pp.from_json(network_path)
    if load_name:
        net.load.loc[net.load.name == load_name, "p_mw"] *= 1 - curtail_percent / 100
    elif bus_index:
        net.load.loc[net.load.bus == int(bus_index), "p_mw"] *= (
            1 - curtail_percent / 100
        )

    pp.to_json(net, network_path)
    record_action(network_path, log)

    return log


def add_battery(network_path: str, bus_index: int, max_energy_kw: float):
    """
    Add a battery to the network.

    Args:
        network_path: the path to the network in editing.
        bus_index: the index of the bus where the battery is to be added.
        max_energy_kw: the maximum energy capacity of the battery in kW.
    """
    msg = f"Add one battery to bus {bus_index} with max energy {max_energy_kw} kW"
    log = {"action": "add_battery", "detail": msg}
    logs = load_action_log(network_path)

    if log in logs:
        return {"action": "None", "detail": "This action has been executed before."}

    net = pp.from_json(network_path)

    pp.create_storage(
        net,
        bus=bus_index,
        p_mw=-max_energy_kw / 1000,
        max_e_mwh=max_energy_kw / 1000,
    )

    pp.to_json(net, network_path)
    record_action(network_path, log)
    return log


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


# ========== PHASE 3: ADVANCED FEATURES ==========


class RepresentationLevel(Enum):
    """Multi-level representation levels for context switching."""

    OVERVIEW = "overview"
    SUMMARY = "summary"
    DETAILED = "detailed"
    GRAPH = "graph"


class ViolationSeverity(Enum):
    """Violation severity levels for intelligent clustering."""

    CRITICAL = "critical"  # >1.15 pu or <0.85 pu voltage, >150% loading
    HIGH = "high"  # >1.1 pu or <0.9 pu voltage, >120% loading
    MEDIUM = "medium"  # >1.05 pu or <0.95 pu voltage, >100% loading
    LOW = "low"  # Minor violations


def get_violation_severity(violation_type: str, value: float) -> ViolationSeverity:
    """Determine violation severity based on type and value."""
    if violation_type == "voltage":
        thresholds = get_voltage_thresholds()
        if value > thresholds.critical_high or value < thresholds.critical_low:
            return ViolationSeverity.CRITICAL
        elif (
            value > thresholds.high_violation_upper
            or value < thresholds.high_violation_lower
        ):
            return ViolationSeverity.HIGH
        elif value > thresholds.v_max or value < thresholds.v_min:
            return ViolationSeverity.MEDIUM
        else:
            return ViolationSeverity.LOW
    elif violation_type == "thermal":
        if value > 150:
            return ViolationSeverity.CRITICAL
        elif value > 120:
            return ViolationSeverity.HIGH
        elif value > 100:
            return ViolationSeverity.MEDIUM
        else:
            return ViolationSeverity.LOW
    return ViolationSeverity.LOW


def cluster_violations_by_proximity(
    net: pp.pandapowerNet, violations: List[Dict], max_distance: int = 3
) -> List[Dict]:
    """Cluster violations by electrical proximity for coordinated planning."""
    if not violations:
        return []

    clusters = []
    visited = set()

    for i, violation in enumerate(violations):
        if i in visited:
            continue

        # Start new cluster
        cluster = {
            "cluster_id": len(clusters),
            "violations": [violation],
            "center_bus": violation.get("bus", violation.get("from_bus")),
            "severity": get_violation_severity(
                "voltage" if "v_mag_pu" in violation else "thermal",
                violation.get("v_mag_pu", violation.get("loading_percent", 0)),
            ).value,
        }
        visited.add(i)

        # Find nearby violations
        center_bus = cluster["center_bus"]
        for j, other_violation in enumerate(violations[i + 1 :], i + 1):
            if j in visited:
                continue

            other_bus = other_violation.get("bus", other_violation.get("from_bus"))
            distance = get_electrical_distance(net, center_bus, other_bus)

            if distance <= max_distance:
                cluster["violations"].append(other_violation)
                visited.add(j)

        clusters.append(cluster)

    # Sort clusters by severity and size
    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    clusters.sort(
        key=lambda c: (severity_order.get(c["severity"], 0), len(c["violations"])),
        reverse=True,
    )

    return clusters


def get_graph_based_representation(net: pp.pandapowerNet) -> Dict[str, Any]:
    """Generate graph-based semantic encoding of network violations and actions."""
    pp.runpp(net)  # Ensure power flow results are available
    # Identify violations and controllable resources
    voltage_violations = []
    thermal_violations = []
    controllable_resources = []

    # Find violations
    thresholds = get_voltage_thresholds()
    for idx, (i, row) in enumerate(net.bus.iterrows()):
        v_mag = net.res_bus.vm_pu.iloc[idx]
        if v_mag > thresholds.v_max or v_mag < thresholds.v_min:
            voltage_violations.append(
                {
                    "id": f"viol_v_{i}",
                    "type": "voltage_violation",
                    "bus": i,
                    "value": round(float(v_mag), 3),
                    "severity": get_violation_severity("voltage", v_mag).value,
                }
            )

    for idx, (i, row) in enumerate(net.line.iterrows()):
        loading = net.res_line.loading_percent.iloc[idx]
        if loading > 100:
            thermal_violations.append(
                {
                    "id": f"viol_t_line_{i}",
                    "type": "thermal_violation",
                    "name": row["name"],
                    "from_bus": int(row["from_bus"]),
                    "to_bus": int(row["to_bus"]),
                    "value": round(float(loading), 3),
                    "severity": get_violation_severity("thermal", loading).value,
                }
            )

    for idx, (i, row) in enumerate(net.trafo.iterrows()):
        loading = net.res_trafo.loading_percent.iloc[idx]
        if loading > 100:
            thermal_violations.append(
                {
                    "id": f"viol_t_trafo_{i}",
                    "type": "thermal_violation",
                    "name": row["name"],
                    "from_bus": int(row["hv_bus"]),
                    "to_bus": int(row["lv_bus"]),
                    "value": round(float(loading), 3),
                    "severity": get_violation_severity("thermal", loading).value,
                }
            )

    # Find controllable resources
    for i, row in net.load.iterrows():
        if row.get("curtailable", False):
            controllable_resources.append(
                {
                    "id": f"ctrl_load_{i}",
                    "type": "curtailable_load",
                    "bus": int(row["bus"]),
                    "name": row["name"],
                    "capacity": round(float(row["p_mw"]), 3),
                    "actions": ["curtail_load"],
                }
            )

    for i, row in net.switch.iterrows():
        controllable_resources.append(
            {
                "id": f"ctrl_switch_{i}",
                "type": "switch",
                "bus": int(row["bus"]),
                "name": row["name"],
                "state": "closed" if row["closed"] else "open",
                "actions": ["update_switch_status"],
            }
        )

    # Add potential battery locations (buses with violations)
    violation_buses = set()
    for v in voltage_violations:
        violation_buses.add(v["bus"])
    for v in thermal_violations:
        violation_buses.add(v["from_bus"])
        violation_buses.add(v["to_bus"])

    for bus in violation_buses:
        controllable_resources.append(
            {
                "id": f"ctrl_battery_{bus}",
                "type": "battery_location",
                "bus": bus,
                "capacity": 1000,  # kW
                "actions": ["add_battery"],
            }
        )

    # Build action graph - connections between violations and nearby resources
    action_edges = []
    all_violations = voltage_violations + thermal_violations

    for violation in all_violations:
        viol_bus = violation.get("bus", violation.get("from_bus"))

        for resource in controllable_resources:
            resource_bus = resource["bus"]
            distance = get_electrical_distance(net, viol_bus, resource_bus)

            if distance <= 5:  # Within 5 hops
                action_edges.append(
                    {
                        "from": violation["id"],
                        "to": resource["id"],
                        "distance": distance,
                        "effectiveness": max(
                            0.1, 1.0 - (distance * 0.2)
                        ),  # Decreases with distance
                    }
                )

    return {
        "graph_representation": {
            "violations": all_violations,
            "controllable_resources": controllable_resources,
            "action_edges": action_edges,
            "violation_clusters": cluster_violations_by_proximity(net, all_violations),
        },
        "meta": {
            "total_violations": len(all_violations),
            "total_resources": len(controllable_resources),
            "total_connections": len(action_edges),
        },
    }


def get_adaptive_field_names(network_size: int) -> Dict[str, str]:
    """Get optimized field names based on network size."""
    if network_size > 1000:
        # Ultra-compressed for very large networks
        return {
            "v_mag_pu": "v",
            "loading_percent": "ld",
            "from_bus": "fr",
            "to_bus": "to",
            "curtailable": "ct",
            "p_mw": "p",
            "q_mvar": "q",
            "v_angle_degree": "ang",
            "closed": "cl",
        }
    elif network_size > 500:
        # Moderately compressed
        return {
            "v_mag_pu": "v_pu",
            "loading_percent": "load_pct",
            "from_bus": "from_bus",
            "to_bus": "to_bus",
            "curtailable": "curt",
            "p_mw": "p_mw",
            "q_mvar": "q_mvar",
            "v_angle_degree": "v_ang",
            "closed": "closed",
        }
    else:
        # Standard field names for small networks
        return {}


def apply_dynamic_compression(
    data: Dict[str, Any], network_size: int
) -> Dict[str, Any]:
    """Apply dynamic compression based on network size."""
    field_mapping = get_adaptive_field_names(network_size)

    if not field_mapping:
        return data

    def compress_dict(obj):
        if isinstance(obj, dict):
            return {field_mapping.get(k, k): compress_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [compress_dict(item) for item in obj]
        else:
            return obj

    return compress_dict(data)


def get_multi_level_representation(
    net: pp.pandapowerNet, level: RepresentationLevel = RepresentationLevel.SUMMARY
) -> Dict[str, Any]:
    """Get multi-level network representation with context switching."""
    pp.runpp(net)  # Ensure power flow results are available
    total_buses = len(net.bus)

    if level == RepresentationLevel.OVERVIEW:
        # Ultra-compact overview (50-100 tokens)
        violations = []
        thresholds = get_voltage_thresholds()
        for idx, (i, row) in enumerate(net.bus.iterrows()):
            v_mag = net.res_bus.vm_pu.iloc[idx]
            if v_mag > thresholds.v_max or v_mag < thresholds.v_min:
                violations.append({"bus": i, "v": round(float(v_mag), 2)})

        thermal_issues = 0
        for idx, (i, row) in enumerate(net.line.iterrows()):
            if net.res_line.loading_percent.iloc[idx] > 100:
                thermal_issues += 1

        return {
            "level": "overview",
            "buses": total_buses,
            "v_violations": len(violations),
            "thermal_violations": thermal_issues,
            "critical_buses": [
                v["bus"] for v in violations if v["v"] > 1.1 or v["v"] < 0.9
            ],
        }

    elif level == RepresentationLevel.SUMMARY:
        # Compact summary (200-500 tokens)
        return get_network_status(net, violations_only=True)

    elif level == RepresentationLevel.DETAILED:
        # Detailed view (1000+ tokens)
        return get_hierarchical_network_status(net)

    elif level == RepresentationLevel.GRAPH:
        # Graph-based representation
        return get_graph_based_representation(net)

    return get_network_status(net)


def get_advanced_network_status(
    network, max_tokens: Optional[int] = None, context_mode: str = "adaptive"
) -> Dict[str, Any]:
    """
    Phase 3: Advanced network status with intelligent format selection.

    Args:
        network: pandapower network object
        max_tokens: maximum token budget (auto-selects appropriate level)
        context_mode: "adaptive", "graph", "hierarchical", "violations_only"

    Returns:
        Optimally formatted network status
    """
    net = network
    pp.runpp(net)

    total_buses = len(net.bus)

    # Auto-select representation level based on constraints
    if max_tokens:
        if max_tokens < 200:
            level = RepresentationLevel.OVERVIEW
        elif max_tokens < 1000:
            level = RepresentationLevel.SUMMARY
        elif context_mode == "graph":
            level = RepresentationLevel.GRAPH
        else:
            level = RepresentationLevel.DETAILED
    else:
        # Default adaptive selection
        if total_buses > 2000:
            level = RepresentationLevel.OVERVIEW
        elif total_buses > 500:
            level = RepresentationLevel.SUMMARY
        elif context_mode == "graph":
            level = RepresentationLevel.GRAPH
        else:
            level = RepresentationLevel.DETAILED

    # Get representation
    if context_mode == "graph" or level == RepresentationLevel.GRAPH:
        result = get_graph_based_representation(net)
    else:
        result = get_multi_level_representation(net, level)

    # Apply dynamic compression for large networks
    if total_buses > 500:
        result = apply_dynamic_compression(result, total_buses)

    # Add metadata
    result["meta_info"] = {
        "representation_level": level.value,
        "network_size": total_buses,
        "compression_applied": total_buses > 500,
        "context_mode": context_mode,
    }

    return result


# ========== ACTION OPTIMIZATION FUNCTIONS ==========


def find_optimal_switch_configuration(
    net: pp.pandapowerNet, thermal_violations: List[Dict]
) -> List[Dict]:
    """Find optimal switch configuration to resolve multiple thermal violations with minimal actions."""
    if not thermal_violations:
        return []

    # Get current switch states
    switch_states = {}
    for i, row in net.switch.iterrows():
        switch_states[row["name"]] = row["closed"]

    # Find switches that could impact violated lines
    potential_switches = []

    for violation in thermal_violations:
        from_bus = violation.get("from_bus")
        to_bus = violation.get("to_bus")

        # Find switches near the violated line
        for i, switch in net.switch.iterrows():
            switch_bus = switch["bus"]

            # Check if switch is electrically close to violation
            if from_bus and get_electrical_distance(net, switch_bus, from_bus) <= 2:
                potential_switches.append(
                    {
                        "switch_name": switch["name"],
                        "current_state": switch["closed"],
                        "target_violation": violation["name"]
                        if "name" in violation
                        else f"line_{from_bus}_{to_bus}",
                        "impact_score": 1.0
                        / (get_electrical_distance(net, switch_bus, from_bus) + 1),
                    }
                )

    # Sort switches by potential impact
    potential_switches.sort(key=lambda x: x["impact_score"], reverse=True)

    # Generate coordinated switch actions (start with most impactful)
    actions = []
    for switch in potential_switches[:3]:  # Limit to top 3 switches
        # Toggle switch state to potentially reroute power
        new_state = not switch["current_state"]
        actions.append(
            {
                "name": "update_switch_status",
                "args": {"switch_name": switch["switch_name"], "closed": new_state},
                "expected_impact": switch["impact_score"],
            }
        )

    return actions


def find_optimal_battery_placement(
    net: pp.pandapowerNet, voltage_violations: List[Dict], max_batteries: int = 3
) -> List[Dict]:
    """Find optimal battery placement to resolve multiple voltage violations with minimal batteries."""
    if not voltage_violations or max_batteries <= 0:
        return []

    # Calculate central locations that can support multiple violations
    violation_buses = [v["bus"] for v in voltage_violations]

    # Score potential battery locations
    battery_candidates = []

    for bus_id in set(violation_buses):
        # Calculate effectiveness for this location
        total_impact = 0
        supported_violations = 0

        for violation in voltage_violations:
            distance = get_electrical_distance(net, bus_id, violation["bus"])
            if distance <= 3:  # Battery effective within 3 hops
                impact = max(
                    0.1, 1.0 - (distance * 0.3)
                )  # Effectiveness decreases with distance
                total_impact += impact * abs(
                    violation["v_mag_pu"] - 1.0
                )  # Weight by severity
                supported_violations += 1

        if supported_violations > 0:
            battery_candidates.append(
                {
                    "bus": bus_id,
                    "total_impact": total_impact,
                    "supported_violations": supported_violations,
                    "efficiency": total_impact / supported_violations,
                }
            )

    # Sort by efficiency (impact per violation)
    battery_candidates.sort(
        key=lambda x: (x["supported_violations"], x["total_impact"]), reverse=True
    )

    # Select optimal battery placements
    actions = []
    placed_batteries = 0
    covered_violations = set()

    for candidate in battery_candidates:
        if placed_batteries >= max_batteries:
            break

        # Check if this battery would help uncovered violations
        new_coverage = 0
        for violation in voltage_violations:
            if violation["bus"] not in covered_violations:
                distance = get_electrical_distance(
                    net, candidate["bus"], violation["bus"]
                )
                if distance <= 3:
                    new_coverage += 1

        if new_coverage > 0:
            actions.append(
                {
                    "name": "add_battery",
                    "args": {
                        "bus_index": candidate["bus"],
                        "max_energy_kw": 1000,
                        "max_battery_count": 1,
                    },
                    "expected_coverage": new_coverage,
                }
            )

            # Mark violations as covered by this battery
            for violation in voltage_violations:
                distance = get_electrical_distance(
                    net, candidate["bus"], violation["bus"]
                )
                if distance <= 3:
                    covered_violations.add(violation["bus"])

            placed_batteries += 1

    return actions


def find_coordinated_load_curtailment(
    net: pp.pandapowerNet, voltage_violations: List[Dict]
) -> List[Dict]:
    """Find coordinated load curtailment to efficiently resolve voltage violations."""
    if not voltage_violations:
        return []

    # Find curtailable loads near voltage violations
    curtailment_candidates = []

    for i, load in net.load.iterrows():
        if not load.get("curtailable", False):
            continue

        load_bus = load["bus"]
        total_impact = 0
        affected_violations = 0

        for violation in voltage_violations:
            distance = get_electrical_distance(net, load_bus, violation["bus"])
            if distance <= 2:  # Load curtailment effective within 2 hops
                impact = max(0.2, 1.0 - (distance * 0.4))
                total_impact += impact * abs(violation["v_mag_pu"] - 1.0)
                affected_violations += 1

        if affected_violations > 0:
            curtailment_candidates.append(
                {
                    "load_name": load["name"],
                    "bus": load_bus,
                    "current_load": load["p_mw"],
                    "total_impact": total_impact,
                    "affected_violations": affected_violations,
                    "efficiency": total_impact
                    / load["p_mw"],  # Impact per MW curtailed
                }
            )

    # Sort by efficiency (impact per MW)
    curtailment_candidates.sort(key=lambda x: x["efficiency"], reverse=True)

    # Select coordinated curtailment actions
    actions = []
    for candidate in curtailment_candidates[:3]:  # Limit to top 3 loads
        # Calculate appropriate curtailment percentage
        severity_factor = min(candidate["affected_violations"], 3) / 3  # 0 to 1
        curtail_percentage = min(25, 5 + (severity_factor * 20))  # 5% to 25%

        actions.append(
            {
                "name": "curtail_load",
                "args": {
                    "load_name": candidate["load_name"],
                    "curtail_percent": curtail_percentage,
                },
                "expected_impact": candidate["total_impact"],
            }
        )

    return actions


def generate_optimized_action_plan(
    net: pp.pandapowerNet, violations: Dict[str, List]
) -> List[Dict]:
    """Generate an optimized action plan that minimizes the number of actions needed."""
    voltage_violations = violations.get("voltage", [])
    thermal_violations = violations.get("thermal", [])

    optimized_actions = []

    # 1. Prioritize switch optimizations for thermal violations (highest impact)
    if thermal_violations:
        switch_actions = find_optimal_switch_configuration(net, thermal_violations)
        optimized_actions.extend(switch_actions)

    # 2. Optimal battery placement for voltage violations
    if voltage_violations:
        battery_actions = find_optimal_battery_placement(
            net, voltage_violations, max_batteries=3
        )
        optimized_actions.extend(battery_actions)

    # 3. Coordinated load curtailment as last resort
    remaining_voltage_violations = voltage_violations
    if (
        len(optimized_actions) == 0 or len(voltage_violations) > 3
    ):  # If many violations remain
        curtailment_actions = find_coordinated_load_curtailment(
            net, remaining_voltage_violations
        )
        optimized_actions.extend(curtailment_actions)

    # Add priority scores for action ordering
    for i, action in enumerate(optimized_actions):
        action["priority"] = len(optimized_actions) - i
        action["optimization_type"] = "coordinated"

    return optimized_actions


def get_optimized_network_status(
    net: pp.pandapowerNet, context_mode: str = "optimization"
) -> Dict[str, Any]:
    """Get network status with optimization-focused information."""
    pp.runpp(net)

    # Get basic violations
    voltage_violations = []
    thermal_violations = []

    for idx, (i, row) in enumerate(net.bus.iterrows()):
        v_mag = net.res_bus.vm_pu.iloc[idx]
        thresholds = get_voltage_thresholds()
        if v_mag > thresholds.v_max or v_mag < thresholds.v_min:
            voltage_violations.append(
                {
                    "bus": i,
                    "name": row["name"],
                    "v_mag_pu": round(float(v_mag), 3),
                    "severity": get_violation_severity("voltage", v_mag).value,
                    "deviation": abs(v_mag - 1.0),
                }
            )

    for idx, (i, row) in enumerate(net.line.iterrows()):
        loading = net.res_line.loading_percent.iloc[idx]
        if loading > 100:
            thermal_violations.append(
                {
                    "name": row["name"],
                    "loading_percent": round(float(loading), 3),
                    "from_bus": int(row["from_bus"]),
                    "to_bus": int(row["to_bus"]),
                    "severity": get_violation_severity("thermal", loading).value,
                    "overload": loading - 100,
                }
            )

    # Generate optimized action plan
    violations = {"voltage": voltage_violations, "thermal": thermal_violations}
    optimized_plan = generate_optimized_action_plan(net, violations)

    # Get available resources for context
    curtailable_loads = []
    for i, row in net.load.iterrows():
        if row.get("curtailable", False):
            curtailable_loads.append(
                {
                    "name": row["name"],
                    "bus": int(row["bus"]),
                    "p_mw": round(float(row["p_mw"]), 3),
                }
            )

    switches = []
    for i, row in net.switch.iterrows():
        switches.append(
            {"name": row["name"], "bus": int(row["bus"]), "closed": bool(row["closed"])}
        )

    return {
        "optimization_context": {
            "total_violations": len(voltage_violations) + len(thermal_violations),
            "voltage_violations": voltage_violations,
            "thermal_violations": thermal_violations,
            "optimized_action_plan": optimized_plan,
            "total_optimized_actions": len(optimized_plan),
            "action_efficiency": len(voltage_violations + thermal_violations)
            / max(1, len(optimized_plan)),
        },
        "available_resources": {
            "curtailable_loads": curtailable_loads,
            "switches": switches,
            "max_batteries": 3,
        },
        "coordination_opportunities": {
            "violation_clusters": cluster_violations_by_proximity(
                net, voltage_violations + thermal_violations
            ),
            "multi_violation_switches": [
                s
                for s in switches
                if any(
                    get_electrical_distance(
                        net, s["bus"], v.get("from_bus", v.get("bus"))
                    )
                    <= 2
                    for v in thermal_violations + voltage_violations
                )
            ],
        },
    }
