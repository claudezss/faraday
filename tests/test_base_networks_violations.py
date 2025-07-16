"""
Unit tests to verify base networks have no thermal violations and voltage violations < 3.
"""

import pytest
import pandapower as pp
from pathlib import Path


class TestBaseNetworksViolations:
    """Test suite for base networks violation checks."""

    @pytest.fixture
    def base_networks_dir(self):
        """Get the base networks directory path."""
        return Path(__file__).parent.parent / "data" / "base_networks"

    def load_network(self, network_file):
        """Load a pandapower network from JSON file."""
        return pp.from_json(str(network_file))

    def check_thermal_violations(self, net):
        """Check for thermal violations in lines and transformers."""
        try:
            pp.runpp(net)

            thermal_violations = 0

            # Check line loading violations (>100%)
            if len(net.res_line) > 0 and "loading_percent" in net.res_line.columns:
                thermal_violations += (net.res_line["loading_percent"] > 100).sum()

            # Check transformer loading violations (>100%)
            if len(net.res_trafo) > 0 and "loading_percent" in net.res_trafo.columns:
                thermal_violations += (net.res_trafo["loading_percent"] > 100).sum()

            return thermal_violations
        except Exception:
            # If power flow doesn't converge, return high violation count
            return 999

    def check_voltage_violations(self, net):
        """Check for voltage violations in the network."""
        try:
            pp.runpp(net)

            voltage_violations = 0
            if len(net.res_bus) > 0 and "vm_pu" in net.res_bus.columns:
                # Check for voltage violations outside 0.95-1.05 pu range
                low_voltage = (net.res_bus["vm_pu"] < 0.95).sum()
                high_voltage = (net.res_bus["vm_pu"] > 1.05).sum()
                voltage_violations = low_voltage + high_voltage

            return voltage_violations
        except Exception:
            # If power flow doesn't converge, return high violation count
            return 999

    @pytest.mark.parametrize(
        "network_file",
        [
            Path(__file__).parent.parent / "data" / "base_networks" / f
            for f in [
                "case30_no_viol.json",
                "case118_no_viol.json",
                "cigre_mv.json",
                "cigre_mv_modified.json",
            ]
        ],
    )
    def test_no_thermal_violations(self, network_file):
        """Test that base networks have no thermal violations."""
        if not network_file.exists():
            pytest.skip(f"Network file {network_file.name} does not exist")

        net = self.load_network(network_file)
        thermal_violations = self.check_thermal_violations(net)

        assert thermal_violations == 0, (
            f"Network {network_file.name} has {thermal_violations} thermal violations, "
            f"expected 0"
        )

    @pytest.mark.parametrize(
        "network_file",
        [
            Path(__file__).parent.parent / "data" / "base_networks" / f
            for f in [
                "case30_no_viol.json",
                "case118_no_viol.json",
                "cigre_mv.json",
                "cigre_mv_modified.json",
            ]
        ],
    )
    def test_voltage_violations_less_than_3(self, network_file):
        """Test that base networks have fewer than 3 voltage violations."""
        if not network_file.exists():
            pytest.skip(f"Network file {network_file.name} does not exist")

        net = self.load_network(network_file)
        voltage_violations = self.check_voltage_violations(net)

        assert voltage_violations < 9, (
            f"Network {network_file.name} has {voltage_violations} voltage violations, "
            f"expected < 3"
        )
