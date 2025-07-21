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
from faraday.utils import fill_missing_names


class NetworkGenerator:
    """Generates test networks with named elements and solvable violations."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional random seed."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def add_violations(
        self, net: pp.pandapowerNet, severity: str = "medium"
    ) -> pp.pandapowerNet:
        """Add thermal violations by gradually increasing max 3 loads until severity is met.

        Args:
            net: Pandapower network to modify
            severity: Violation severity level ("light", "medium", "severe")

        Returns:
            Modified network with thermal violations validated by powerflow
        """

        # Define thermal violation thresholds for different severities
        severity_thresholds = {
            "light": 110,  # 110% line loading
            "medium": 130,  # 130% line loading
            "severe": 150,  # 150% line loading
        }

        target_loading = severity_thresholds.get(
            severity, severity_thresholds["medium"]
        )

        # Get all loads and select up to 3 randomly
        available_loads = net.load.index.tolist()
        if len(available_loads) == 0:
            print("Warning: No loads found in network")
            return net

        # Limit to maximum 3 loads
        num_loads_to_modify = min(3, len(available_loads))
        selected_loads = random.sample(available_loads, num_loads_to_modify)

        print(
            f"Creating thermal violations with {severity} severity (target: {target_loading}% loading)"
        )
        print(f"Modifying {num_loads_to_modify} loads gradually")

        # Store original values for rollback if needed
        original_p_mw = {}

        for load_idx in selected_loads:
            original_p_mw[load_idx] = net.load.loc[load_idx, "p_mw"]
            # Mark as curtailable for AI to fix
            net.load.loc[load_idx, "curtailable"] = True

        # Gradually increase loads until thermal violations meet severity
        increment = 0.5  # 10% increment per iteration
        max_iterations = 50  # Safety limit
        iteration = 0
        violations_achieved = False

        while not violations_achieved and iteration < max_iterations:
            iteration += 1

            # Increase selected loads by increment
            for load_idx in selected_loads:
                net.load.loc[load_idx, "p_mw"] = original_p_mw[load_idx] * (
                    1 + increment * iteration
                )

            # Run powerflow to check thermal violations
            try:
                pp.runpp(net)

                # Check thermal violations
                thermal_violations = []
                max_loading = 0

                for line_idx, loading in zip(
                    net.line.index.tolist() + net.trafo.index.tolist(),
                    net.res_line.loading_percent.tolist()
                    + net.res_trafo.loading_percent.tolist(),
                ):
                    if loading > 100:  # Any overloading
                        thermal_violations.append(
                            {"line": line_idx, "loading_percent": loading}
                        )
                        max_loading = max(max_loading, loading)

                # Check if we've achieved target severity
                if len(thermal_violations) > 0 and max_loading >= target_loading:
                    violations_achieved = True
                    print(
                        f"✓ Successfully created {len(thermal_violations)} thermal violations after {iteration} iterations:"
                    )

                    for viol in thermal_violations:
                        line_name = (
                            net.line.loc[viol["line"], "name"]
                            if "name" in net.line.columns
                            else f"Line_{viol['line']}"
                        )
                        print(
                            f"  - {line_name}: {viol['loading_percent']:.1f}% loading"
                        )

                    # Show final load modifications
                    for load_idx in selected_loads:
                        load_name = (
                            net.load.loc[load_idx, "name"]
                            if "name" in net.load.columns
                            else f"Load_{load_idx}"
                        )
                        final_multiplier = 1 + increment * iteration
                        print(
                            f"  - Modified {load_name}: P={original_p_mw[load_idx]:.2f} -> {net.load.loc[load_idx, 'p_mw']:.2f} MW (x{final_multiplier:.2f})"
                        )

                    print(f"  - Maximum line loading: {max_loading:.1f}%")

                elif iteration % 10 == 0:  # Progress update every 10 iterations
                    current_max = max_loading if thermal_violations else 0
                    print(
                        f"  Iteration {iteration}: Max loading = {current_max:.1f}%, target = {target_loading}%"
                    )

            except Exception as e:
                print(f"Error during powerflow at iteration {iteration}: {e}")
                # Rollback to previous state
                for load_idx in selected_loads:
                    net.load.loc[load_idx, "p_mw"] = original_p_mw[load_idx] * (
                        1 + increment * (iteration - 1)
                    )
                break

        if not violations_achieved:
            print(
                f"Warning: Could not achieve {severity} thermal violations after {max_iterations} iterations"
            )
            print(
                "Network may be too robust or loads insufficient to create thermal violations"
            )
            # Keep the final state even if target not achieved

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

    def generate_test_network(
        self,
        base_net: pp.pandapowerNet,
        severity: str = "medium",
        add_switches: bool = False,
    ) -> pp.pandapowerNet:
        """Generate a complete test network with violations and solution capabilities."""

        # Make a copy to avoid modifying original
        net = deepcopy(base_net)

        # Fill missing names
        net = fill_missing_names(net)

        # Add violations
        net = self.add_violations(net, severity)

        # Add solution capabilities
        if add_switches:
            net = self.add_switches_for_topology_control(net)

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
            for line_idx, loading in zip(
                net.line.index,
                net.res_line.loading_percent.tolist()
                + net.res_trafo.loading_percent.tolist()
                if len(net.trafo) > 0
                else [],
            ):
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
faraday {network_file}
```
"""

        with open(output_dir / "README.md", "w") as f:
            f.write(readme_content)

        return validation
