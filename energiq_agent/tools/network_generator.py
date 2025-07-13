"""
Network generator utility for creating test networks with violations.
Fills missing names and adds solvable violations to pandapower networks.
"""

from copy import deepcopy

import pandapower as pp
import numpy as np
import random
from typing import Dict, Optional
from pathlib import Path
import json
import pandas as pd


class NetworkGenerator:
    """Generates test networks with named elements and solvable violations."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional random seed."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def fill_missing_names(self, net: pp.pandapowerNet) -> pp.pandapowerNet:
        """Fill missing names for loads, lines, switches, and transformers."""

        # Initialize curtailable column if it doesn't exist
        if "curtailable" not in net.load.columns:
            net.load["curtailable"] = False

        # Fill load names
        for idx, row in net.load.iterrows():
            if pd.isna(row.get("name")) or row.get("name") == "":
                bus_name = (
                    net.bus.loc[row.bus, "name"]
                    if "name" in net.bus.columns
                    else f"Bus_{row.bus}"
                )
                load_type = row.get("type", "load")
                net.load.loc[idx, "name"] = f"Load_{bus_name}_{load_type}_{idx}"

        # Fill gen names
        for idx, row in net.gen.iterrows():
            if pd.isna(row.get("name")) or row.get("name") == "":
                bus_name = (
                    net.bus.loc[row.bus, "name"]
                    if "name" in net.bus.columns
                    else f"Bus_{row.bus}"
                )
                net.gen.loc[idx, "name"] = f"Gen_{bus_name}_{idx}"

        # Fill sgen names
        for idx, row in net.sgen.iterrows():
            if pd.isna(row.get("name")) or row.get("name") == "":
                bus_name = (
                    net.bus.loc[row.bus, "name"]
                    if "name" in net.bus.columns
                    else f"Bus_{row.bus}"
                )
                net.sgen.loc[idx, "name"] = f"SyncGen_{bus_name}_{idx}"

        # Fill line names
        for idx, row in net.line.iterrows():
            if pd.isna(row.get("name")) or row.get("name") == "":
                from_bus = (
                    net.bus.loc[row.from_bus, "name"]
                    if "name" in net.bus.columns
                    else f"Bus_{row.from_bus}"
                )
                to_bus = (
                    net.bus.loc[row.to_bus, "name"]
                    if "name" in net.bus.columns
                    else f"Bus_{row.to_bus}"
                )
                net.line.loc[idx, "name"] = f"Line_{from_bus}_to_{to_bus}_{idx}"

        # Fill transformer names
        if len(net.trafo) > 0:
            for idx, row in net.trafo.iterrows():
                if pd.isna(row.get("name")) or row.get("name") == "":
                    hv_bus = (
                        net.bus.loc[row.hv_bus, "name"]
                        if "name" in net.bus.columns
                        else f"Bus_{row.hv_bus}"
                    )
                    lv_bus = (
                        net.bus.loc[row.lv_bus, "name"]
                        if "name" in net.bus.columns
                        else f"Bus_{row.lv_bus}"
                    )
                    net.trafo.loc[idx, "name"] = f"Trafo_{hv_bus}_to_{lv_bus}_{idx}"

        # Fill switch names
        if len(net.switch) > 0:
            for idx, row in net.switch.iterrows():
                if pd.isna(row.get("name")) or row.get("name") == "":
                    bus_name = (
                        net.bus.loc[row.bus, "name"]
                        if "name" in net.bus.columns
                        else f"Bus_{row.bus}"
                    )
                    element_type = row.et
                    element_idx = row.element
                    net.switch.loc[idx, "name"] = (
                        f"Switch_{bus_name}_{element_type}_{element_idx}_{idx}"
                    )

        # Fill bus names if missing
        for idx, row in net.bus.iterrows():
            if pd.isna(row.get("name")) or row.get("name") == "":
                net.bus.loc[idx, "name"] = f"Bus_{idx}"

        return net

    def add_voltage_violations(
        self, net: pp.pandapowerNet, violation_count: int = 3, severity: str = "medium"
    ) -> pp.pandapowerNet:
        """Add voltage violations by modifying loads and generation."""

        severity_multipliers = {
            "light": (
                0.96,
                1.04,
                1.15,
            ),  # (min_v, max_v, load_multiplier) - more conservative
            "medium": (0.92, 1.07, 1.3),
            "severe": (0.88, 1.10, 1.6),
        }

        min_v, max_v, load_mult = severity_multipliers.get(
            severity, severity_multipliers["medium"]
        )

        # Get buses that can have violations (excluding slack bus)
        candidate_buses = net.bus[net.bus.type != "slack"].index.tolist()

        if len(candidate_buses) < violation_count:
            violation_count = len(candidate_buses)

        violation_buses = random.sample(candidate_buses, violation_count)

        for i, bus_idx in enumerate(violation_buses):
            # Only create undervoltage violations (more solvable)
            # Increase load to cause voltage drop
            loads_at_bus = net.load[net.load.bus == bus_idx]
            if len(loads_at_bus) > 0:
                for load_idx in loads_at_bus.index:
                    # Conservative load increase
                    net.load.loc[load_idx, "p_mw"] *= load_mult
                    net.load.loc[load_idx, "q_mvar"] *= load_mult
                    # Mark as curtailable for AI to fix
                    net.load.loc[load_idx, "curtailable"] = True
            else:
                # Create new moderate load
                pp.create_load(
                    net,
                    bus=bus_idx,
                    p_mw=1.0 * load_mult,  # More conservative
                    q_mvar=0.3 * load_mult,
                    name=f"High_Load_Bus_{bus_idx}",
                    curtailable=True,
                )

        return net

    def add_thermal_violations(
        self, net: pp.pandapowerNet, violation_count: int = 2, severity: str = "medium"
    ) -> pp.pandapowerNet:
        """Add thermal violations by overloading lines."""

        severity_multipliers = {
            "light": 1.1,  # 110% loading
            "medium": 1.3,  # 130% loading
            "severe": 1.6,  # 160% loading
        }

        loading_mult = severity_multipliers.get(
            severity, severity_multipliers["medium"]
        )

        # Select random lines to overload
        candidate_lines = net.line.index.tolist()

        if len(candidate_lines) < violation_count:
            violation_count = len(candidate_lines)

        violation_lines = random.sample(candidate_lines, violation_count)

        for line_idx in violation_lines:
            line = net.line.loc[line_idx]
            to_bus = line.to_bus

            # Increase load at the end of the line
            loads_at_bus = net.load[net.load.bus == to_bus]
            if len(loads_at_bus) > 0:
                for load_idx in loads_at_bus.index:
                    net.load.loc[load_idx, "p_mw"] *= loading_mult
                    net.load.loc[load_idx, "q_mvar"] *= loading_mult
                    net.load.loc[load_idx, "curtailable"] = True
            else:
                # Create new load to overload the line
                pp.create_load(
                    net,
                    bus=to_bus,
                    p_mw=1.5 * loading_mult,
                    q_mvar=0.4 * loading_mult,
                    name=f"Overload_Bus_{to_bus}",
                    curtailable=True,
                )

        return net

    def add_switches_for_topology_control(
        self, net: pp.pandapowerNet
    ) -> pp.pandapowerNet:
        """Add switches to enable topology control solutions."""

        # Add bus-to-bus switches for network reconfiguration
        buses_with_multiple_connections = []

        for bus_idx in net.bus.index:
            connected_lines = len(
                net.line[(net.line.from_bus == bus_idx) | (net.line.to_bus == bus_idx)]
            )
            if connected_lines >= 2:
                buses_with_multiple_connections.append(bus_idx)

        # Create normally open switches between some buses
        for i in range(min(3, len(buses_with_multiple_connections) - 1)):
            bus1 = buses_with_multiple_connections[i]
            bus2 = buses_with_multiple_connections[i + 1]

            # Check if switch doesn't already exist
            existing_switch = net.switch[
                (net.switch.bus == bus1)
                & (net.switch.element == bus2)
                & (net.switch.et == "b")
            ]

            if len(existing_switch) == 0:
                pp.create_switch(
                    net,
                    bus=bus1,
                    element=bus2,
                    et="b",
                    closed=False,  # Normally open for reconfiguration
                    name=f"Reconfig_Switch_{bus1}_{bus2}",
                )

        return net

    def add_battery_opportunities(self, net: pp.pandapowerNet) -> pp.pandapowerNet:
        """Mark suitable buses for battery placement."""

        # Add metadata to buses to indicate battery suitability
        net.bus["battery_suitable"] = False

        # Buses at the end of feeders are good for batteries
        for bus_idx in net.bus.index:
            connected_lines = len(
                net.line[(net.line.from_bus == bus_idx) | (net.line.to_bus == bus_idx)]
            )

            # End buses (only one connection) or important junction buses
            if connected_lines == 1 or connected_lines >= 3:
                net.bus.loc[bus_idx, "battery_suitable"] = True

        return net

    def generate_test_network(
        self,
        base_net: pp.pandapowerNet,
        voltage_violations: int = 2,
        thermal_violations: int = 1,
        severity: str = "medium",
        add_switches: bool = True,
        add_battery_sites: bool = True,
    ) -> pp.pandapowerNet:
        """Generate a complete test network with violations and solution capabilities."""

        # Make a copy to avoid modifying original
        net = deepcopy(base_net)

        # Fill missing names
        net = self.fill_missing_names(net)

        # Add violations
        if voltage_violations > 0:
            net = self.add_voltage_violations(net, voltage_violations, severity)

        if thermal_violations > 0:
            net = self.add_thermal_violations(net, thermal_violations, severity)

        # Add solution capabilities
        if add_switches:
            net = self.add_switches_for_topology_control(net)

        if add_battery_sites:
            net = self.add_battery_opportunities(net)

        return net

    def validate_network(self, net: pp.pandapowerNet) -> Dict:
        """Validate the network and return violation statistics."""

        try:
            pp.runpp(net)

            # Count violations
            voltage_violations = []
            thermal_violations = []

            # Check voltage violations
            for bus_idx, vm_pu in zip(net.bus.index, net.res_bus.vm_pu):
                if vm_pu > 1.05 or vm_pu < 0.95:
                    violation_type = "overvoltage" if vm_pu > 1.05 else "undervoltage"
                    voltage_violations.append(
                        {"bus": bus_idx, "voltage_pu": vm_pu, "type": violation_type}
                    )

            # Check thermal violations
            for line_idx, loading in zip(net.line.index, net.res_line.loading_percent):
                if loading > 100:
                    thermal_violations.append(
                        {"line": line_idx, "loading_percent": loading}
                    )

            return {
                "converged": True,
                "voltage_violations": voltage_violations,
                "thermal_violations": thermal_violations,
                "total_violations": len(voltage_violations) + len(thermal_violations),
                "curtailable_loads": int(net.load.get("curtailable", False).sum()),
                "controllable_dg": int(net.sgen.get("controllable", False).sum())
                if len(net.sgen) > 0
                else 0,
                "switches": len(net.switch),
                "battery_sites": int(net.bus.get("battery_suitable", False).sum())
                if "battery_suitable" in net.bus.columns
                else 0,
            }

        except Exception as e:
            return {
                "converged": False,
                "error": str(e),
                "voltage_violations": [],
                "thermal_violations": [],
                "total_violations": 0,
            }

    def save_network(self, net: pp.pandapowerNet, output_dir: Path, name: str):
        """Save the network and create metadata."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save network
        network_file = output_dir / "net.json"
        pp.to_json(net, str(network_file))

        # Validate and save metadata
        validation = self.validate_network(net)

        metadata = {
            "name": name,
            "description": f"Test network with {validation['total_violations']} violations",
            "buses": len(net.bus),
            "lines": len(net.line),
            "loads": len(net.load),
            "generators": len(net.sgen) if len(net.sgen) > 0 else 0,
            "transformers": len(net.trafo) if len(net.trafo) > 0 else 0,
            "switches": len(net.switch) if len(net.switch) > 0 else 0,
            "validation": validation,
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Create README
        readme_content = f"""# {name}

{metadata["description"]}

## Network Statistics
- **Buses**: {metadata["buses"]}
- **Lines**: {metadata["lines"]}
- **Loads**: {metadata["loads"]}
- **Generators**: {metadata["generators"]}
- **Transformers**: {metadata["transformers"]}
- **Switches**: {metadata["switches"]}

## Violations
- **Total**: {validation["total_violations"]}
- **Voltage**: {len(validation["voltage_violations"])}
- **Thermal**: {len(validation["thermal_violations"])}

## Solution Capabilities
- **Curtailable Loads**: {validation["curtailable_loads"]}
- **Controllable DG**: {validation["controllable_dg"]}
- **Topology Switches**: {validation["switches"]}
- **Battery Sites**: {validation["battery_sites"]}

## Usage

```bash
energiq-agent {network_file}
```
"""

        with open(output_dir / "README.md", "w") as f:
            f.write(readme_content)

        return validation
