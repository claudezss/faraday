"""
Data Processing Utilities

Network data processing and analysis utilities for the dashboard.
"""

from typing import Dict, Any, List
import pandapower as pp


from faraday.dashboard.utils.session_state import SessionStateManager
from faraday.tools.pandapower import get_violations, get_voltage_thresholds


class NetworkDataProcessor:
    """Network data processing and analysis utilities."""

    def __init__(self):
        self.thresholds = get_voltage_thresholds()

    def get_network_metrics(self, network_type: str = "current") -> Dict[str, Any]:
        """Get comprehensive network metrics."""
        try:
            net = self._get_network_by_type(network_type)
            if net is None:
                return {}

            # Run power flow analysis
            pp.runpp(net)

            # Basic network structure
            metrics = {
                "total_buses": len(net.bus),
                "total_lines": len(net.line),
                "total_transformers": len(net.trafo)
                if hasattr(net, "trafo") and not net.trafo.empty
                else 0,
                "total_switches": len(net.switch)
                if hasattr(net, "switch") and not net.switch.empty
                else 0,
                "total_loads": len(net.load),
                "total_generators": len(net.sgen)
                if hasattr(net, "sgen") and not net.sgen.empty
                else 0,
            }

            # Power metrics
            metrics.update(
                {
                    "total_load_mw": net.load.p_mw.sum() if not net.load.empty else 0,
                    "total_generation_mw": net.sgen.p_mw.sum()
                    if hasattr(net, "sgen") and not net.sgen.empty
                    else 0,
                    "total_load_mvar": net.load.q_mvar.sum()
                    if not net.load.empty
                    else 0,
                    "total_generation_mvar": net.sgen.q_mvar.sum()
                    if hasattr(net, "sgen") and not net.sgen.empty
                    else 0,
                }
            )

            # Power balance
            metrics["power_balance"] = (
                metrics["total_generation_mw"] - metrics["total_load_mw"]
            )

            # Voltage metrics
            if hasattr(net, "res_bus") and not net.res_bus.empty:
                voltages = net.res_bus.vm_pu.dropna()
                if len(voltages) > 0:
                    metrics.update(
                        {
                            "min_voltage": voltages.min(),
                            "max_voltage": voltages.max(),
                            "avg_voltage": voltages.mean(),
                            "voltage_std": voltages.std(),
                        }
                    )

            # Line loading metrics
            if hasattr(net, "res_line") and not net.res_line.empty:
                loadings = net.res_line.loading_percent.dropna()
                if len(loadings) > 0:
                    metrics.update(
                        {
                            "max_loading": loadings.max(),
                            "avg_loading": loadings.mean(),
                            "loading_std": loadings.std(),
                        }
                    )

            # Power loss calculation
            if hasattr(net, "res_line") and not net.res_line.empty:
                total_losses = (
                    net.res_line.pl_mw.sum() if "pl_mw" in net.res_line.columns else 0
                )
                metrics["power_loss"] = total_losses

            return metrics

        except Exception as e:
            print(f"Error calculating network metrics: {e}")
            return {}

    def get_violations_summary(self, network_type: str = "current") -> Dict[str, Any]:
        """Get violations summary."""
        try:
            net = self._get_network_by_type(network_type)
            if net is None:
                return {
                    "voltage_violations": 0,
                    "thermal_violations": 0,
                    "total_violations": 0,
                    "health_score": 0,
                }

            violations = get_violations(net)

            voltage_count = len(violations.voltage)
            thermal_count = len(violations.thermal)
            disconnected_count = len(violations.disconnected_buses)
            total_count = voltage_count + thermal_count + disconnected_count

            # Calculate health score
            total_elements = len(net.bus) + len(net.line)
            if hasattr(net, "trafo") and not net.trafo.empty:
                total_elements += len(net.trafo)

            health_score = max(0, 100 - (total_count / max(total_elements, 1)) * 100)

            return {
                "voltage_violations": voltage_count,
                "thermal_violations": thermal_count,
                "total_violations": total_count,
                "health_score": health_score,
                "disconnected_buses": len(violations.disconnected_buses),
            }

        except Exception as e:
            print(f"Error calculating violations summary: {e}")
            return {
                "voltage_violations": 0,
                "thermal_violations": 0,
                "total_violations": 0,
                "health_score": 0,
            }

    def get_violations_data(
        self, network_type: str = "current"
    ) -> Dict[str, List[Dict]]:
        """Get detailed violations data."""
        try:
            net = self._get_network_by_type(network_type)
            if net is None:
                return {"voltage_violations": [], "thermal_violations": []}

            violations = get_violations(net)

            # Process voltage violations
            voltage_violations = []
            for v in violations.voltage:
                voltage_violations.append(
                    {
                        "bus_idx": v.bus_idx,
                        "v_mag_pu": v.v_mag_pu,
                        "severity": self._get_voltage_severity(v.v_mag_pu),
                        "deviation": abs(v.v_mag_pu - 1.0),
                    }
                )

            # Process thermal violations
            thermal_violations = []
            for t in violations.thermal:
                thermal_violations.append(
                    {
                        "name": t.name,
                        "from_bus_idx": t.from_bus_idx,
                        "to_bus_idx": t.to_bus_idx,
                        "loading_percent": t.loading_percent,
                        "severity": self._get_thermal_severity(t.loading_percent),
                        "overload": t.loading_percent - 100,
                    }
                )

            return {
                "voltage_violations": voltage_violations,
                "thermal_violations": thermal_violations,
                "disconnected_buses": violations.disconnected_buses,
            }

        except Exception as e:
            print(f"Error getting violations data: {e}")
            return {"voltage_violations": [], "thermal_violations": []}

    def get_controllable_resources(self) -> Dict[str, List[Dict]]:
        """Get available controllable resources."""
        try:
            net = SessionStateManager.get_network()
            if net is None:
                return {
                    "loads": [],
                    "switches": [],
                    "generators": [],
                    "battery_sites": [],
                }

            resources = {
                "loads": [],
                "switches": [],
                "generators": [],
                "battery_sites": [],
            }

            # Curtailable loads
            if not net.load.empty:
                for idx, load in net.load.iterrows():
                    if load.get("curtailable", False):
                        resources["loads"].append(
                            {
                                "index": idx,
                                "name": load.get("name", f"Load_{idx}"),
                                "bus": load["bus"],
                                "p_mw": load["p_mw"],
                                "q_mvar": load["q_mvar"],
                                "curtailable": True,
                            }
                        )

            # Controllable switches
            if hasattr(net, "switch") and not net.switch.empty:
                for idx, switch in net.switch.iterrows():
                    resources["switches"].append(
                        {
                            "index": idx,
                            "name": switch.get("name", f"Switch_{idx}"),
                            "bus": switch["bus"],
                            "element": switch["element"],
                            "closed": switch["closed"],
                            "controllable": switch.get("controllable", True),
                        }
                    )

            # Controllable generators
            if hasattr(net, "sgen") and not net.sgen.empty:
                for idx, gen in net.sgen.iterrows():
                    if gen.get("controllable", False):
                        resources["generators"].append(
                            {
                                "index": idx,
                                "name": gen.get("name", f"Gen_{idx}"),
                                "bus": gen["bus"],
                                "p_mw": gen["p_mw"],
                                "q_mvar": gen["q_mvar"],
                                "controllable": True,
                            }
                        )

            # Potential battery sites (all buses)
            for idx, bus in net.bus.iterrows():
                resources["battery_sites"].append(
                    {
                        "bus_index": idx,
                        "name": bus.get("name", f"Bus_{idx}"),
                        "vn_kv": bus["vn_kv"] if "vn_kv" in bus else 1.0,
                        "suitable": True,  # Could add suitability analysis
                    }
                )

            return resources

        except Exception as e:
            print(f"Error getting controllable resources: {e}")
            return {"loads": [], "switches": [], "generators": [], "battery_sites": []}

    def analyze_network_topology(self) -> Dict[str, Any]:
        """Analyze network topology and connectivity."""
        try:
            net = SessionStateManager.get_network()
            if net is None:
                return {}

            analysis = {
                "connectivity": self._analyze_connectivity(net),
                "voltage_levels": self._analyze_voltage_levels(net),
                "zones": self._identify_network_zones(net),
                "critical_elements": self._identify_critical_elements(net),
            }

            return analysis

        except Exception as e:
            print(f"Error in topology analysis: {e}")
            return {}

    def calculate_action_effectiveness(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the potential effectiveness of an action."""
        try:
            net = SessionStateManager.get_network()
            if net is None:
                return {"effectiveness": 0, "risk": "unknown", "impact_areas": []}

            action_type = action.get("name", "")
            args = action.get("args", {})

            effectiveness = {
                "effectiveness": 0,
                "risk": "low",
                "impact_areas": [],
                "side_effects": [],
                "confidence": 0,
            }

            if action_type == "curtail_load":
                effectiveness.update(self._analyze_load_curtailment(net, args))
            elif action_type == "update_switch_status":
                effectiveness.update(self._analyze_switch_operation(net, args))
            elif action_type == "add_battery":
                effectiveness.update(self._analyze_battery_addition(net, args))

            return effectiveness

        except Exception as e:
            print(f"Error calculating action effectiveness: {e}")
            return {"effectiveness": 0, "risk": "unknown", "impact_areas": []}

    def _get_network_by_type(self, network_type: str):
        """Get network by type (current, initial, final)."""
        if network_type == "initial":
            return SessionStateManager.get_initial_network()
        elif network_type == "final":
            return SessionStateManager.get_final_network()
        else:
            return SessionStateManager.get_network()

    def _get_voltage_severity(self, voltage: float) -> str:
        """Determine voltage violation severity."""
        if (
            voltage > self.thresholds.critical_high
            or voltage < self.thresholds.critical_low
        ):
            return "critical"
        elif (
            voltage > self.thresholds.high_violation_upper
            or voltage < self.thresholds.high_violation_lower
        ):
            return "high"
        elif voltage > self.thresholds.v_max or voltage < self.thresholds.v_min:
            return "medium"
        else:
            return "low"

    def _get_thermal_severity(self, loading: float) -> str:
        """Determine thermal violation severity."""
        if loading > 150:
            return "critical"
        elif loading > 120:
            return "high"
        elif loading > 100:
            return "medium"
        else:
            return "low"

    def _analyze_connectivity(self, net) -> Dict[str, Any]:
        """Analyze network connectivity."""
        # Simplified connectivity analysis
        total_buses = len(net.bus)
        total_lines = len(net.line)

        connectivity_ratio = total_lines / max(total_buses, 1)

        return {
            "total_buses": total_buses,
            "total_lines": total_lines,
            "connectivity_ratio": connectivity_ratio,
            "is_well_connected": connectivity_ratio > 1.2,
            "isolated_buses": [],  # Would need graph analysis to find truly isolated buses
        }

    def _analyze_voltage_levels(self, net) -> Dict[str, Any]:
        """Analyze voltage levels in the network."""
        voltage_levels = {}

        if "vn_kv" in net.bus.columns:
            levels = net.bus.vn_kv.unique()
            for level in levels:
                buses_at_level = len(net.bus[net.bus.vn_kv == level])
                voltage_levels[f"{level}_kV"] = buses_at_level

        return voltage_levels

    def _identify_network_zones(self, net) -> List[Dict[str, Any]]:
        """Identify network zones or areas."""
        # Simplified zone identification based on voltage levels
        zones = []

        if "vn_kv" in net.bus.columns:
            for level in net.bus.vn_kv.unique():
                buses_in_zone = net.bus[net.bus.vn_kv == level].index.tolist()
                zones.append(
                    {
                        "zone_id": len(zones),
                        "voltage_level": level,
                        "bus_count": len(buses_in_zone),
                        "buses": buses_in_zone[:10],  # Limit for display
                    }
                )

        return zones

    def _identify_critical_elements(self, net) -> Dict[str, List]:
        """Identify critical network elements."""
        critical = {"buses": [], "lines": [], "transformers": []}

        # Critical buses (high degree, voltage violations)
        try:
            violations = get_violations(net)
            critical["buses"] = [v.bus_idx for v in violations.voltage[:5]]  # Top 5
            critical["lines"] = [t.name for t in violations.thermal[:5]]  # Top 5
        except Exception:
            pass

        return critical

    def _analyze_load_curtailment(self, net, args: Dict) -> Dict[str, Any]:
        """Analyze load curtailment action effectiveness."""
        load_name = args.get("load_name")
        curtail_percent = args.get("curtail_percent", 10)

        effectiveness = 50 + (curtail_percent * 2)  # Simplified calculation

        return {
            "effectiveness": min(100, effectiveness),
            "risk": "low" if curtail_percent < 20 else "medium",
            "impact_areas": [f"Bus connected to {load_name}"],
            "confidence": 80,
        }

    def _analyze_switch_operation(self, net, args: Dict) -> Dict[str, Any]:
        """Analyze switch operation effectiveness."""
        switch_name = args.get("switch_name")
        closed = args.get("closed", True)

        # Switch operations can have high impact but also higher risk
        effectiveness = 75
        risk = "medium" if not closed else "low"  # Opening switches is riskier

        return {
            "effectiveness": effectiveness,
            "risk": risk,
            "impact_areas": [f"Network topology around {switch_name}"],
            "confidence": 70,
        }

    def _analyze_battery_addition(self, net, args: Dict) -> Dict[str, Any]:
        """Analyze battery addition effectiveness."""
        bus_index = args.get("bus_index")
        capacity = args.get("max_energy_kw", 1000)

        # Batteries generally have high effectiveness and low risk
        effectiveness = 60 + (capacity / 100)  # Scales with capacity

        return {
            "effectiveness": min(100, effectiveness),
            "risk": "low",
            "impact_areas": [f"Bus {bus_index} and neighboring buses"],
            "confidence": 85,
        }
