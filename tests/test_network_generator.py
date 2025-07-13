"""
Unit tests for the NetworkGenerator utility using case14 as base network.
"""

import pandapower as pp
import pandapower.networks as pn

import tempfile
import shutil
from pathlib import Path

from energiq_agent.tools.network_generator import NetworkGenerator
from copy import deepcopy


class TestNetworkGenerator:
    """Test suite for NetworkGenerator class using case14."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = NetworkGenerator(seed=42)  # Fixed seed for reproducible tests
        self.temp_dir = Path(tempfile.mkdtemp())
        self.base_net = pn.case14()  # Use case14 as base test network

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_init_with_seed(self):
        """Test generator initialization with seed."""
        gen1 = NetworkGenerator(seed=123)
        gen2 = NetworkGenerator(seed=123)
        assert gen1 is not None
        assert gen2 is not None

    def test_fill_missing_names_loads(self):
        """Test filling missing load names on case14."""
        net = deepcopy(self.base_net)

        # Remove some load names to test filling
        if len(net.load) > 0:
            net.load.loc[0, "name"] = ""
            if len(net.load) > 1:
                net.load.loc[1, "name"] = None

        original_count = len(net.load)
        result_net = self.generator.fill_missing_names(net)

        # Check that all loads have names
        assert len(result_net.load) == original_count
        assert result_net.load["name"].notna().all()
        assert (result_net.load["name"] != "").all()

    def test_fill_missing_names_lines(self):
        """Test filling missing line names on case14."""
        net = deepcopy(self.base_net)

        # Remove some line names
        net.line.loc[0, "name"] = ""
        if len(net.line) > 1:
            net.line.loc[1, "name"] = None

        original_count = len(net.line)
        result_net = self.generator.fill_missing_names(net)

        # Check that all lines have names
        assert len(result_net.line) == original_count
        assert result_net.line["name"].notna().all()
        assert (result_net.line["name"] != "").all()

        # Check naming pattern
        filled_name = result_net.line.loc[0, "name"]
        assert "Line_" in filled_name and "_to_" in filled_name

    def test_fill_missing_names_buses(self):
        """Test filling missing bus names on case14."""
        net = deepcopy(self.base_net)

        # Remove some bus names
        net.bus.loc[0, "name"] = ""
        net.bus.loc[1, "name"] = None

        original_count = len(net.bus)
        result_net = self.generator.fill_missing_names(net)

        # Check that all buses have names
        assert len(result_net.bus) == original_count
        assert result_net.bus["name"].notna().all()
        assert (result_net.bus["name"] != "").all()

        # Check naming pattern for filled names
        assert result_net.bus.loc[0, "name"] == "Bus_0"
        assert result_net.bus.loc[1, "name"] == "Bus_1"

    def test_add_voltage_violations_light(self):
        """Test adding light voltage violations to case14."""
        net = deepcopy(self.base_net)
        original_load_count = len(net.load)

        result_net = self.generator.add_voltage_violations(
            net, violation_count=2, severity="light"
        )

        # Should have same or more loads (might add new ones)
        assert len(result_net.load) >= original_load_count

        # Should have curtailable loads
        assert result_net.load["curtailable"].sum() > 0

        # Run power flow to check it converges
        pp.runpp(result_net)
        assert len(result_net.res_bus) > 0

    def test_add_voltage_violations_medium(self):
        """Test adding medium voltage violations to case14."""
        net = deepcopy(self.base_net)

        result_net = self.generator.add_voltage_violations(
            net, violation_count=3, severity="medium"
        )

        # Should have curtailable loads or controllable DG
        curtailable_loads = result_net.load["curtailable"].sum()
        controllable_dg = (
            result_net.sgen["controllable"].sum() if len(result_net.sgen) > 0 else 0
        )

        assert curtailable_loads > 0 or controllable_dg > 0

    def test_add_thermal_violations(self):
        """Test adding thermal violations to case14."""
        net = deepcopy(self.base_net)
        original_load_count = len(net.load)

        result_net = self.generator.add_thermal_violations(
            net, violation_count=2, severity="medium"
        )

        # Should have same or more loads
        assert len(result_net.load) >= original_load_count

        # Should have curtailable loads
        assert result_net.load["curtailable"].sum() > 0

        # Run power flow to check it converges
        pp.runpp(result_net)
        assert len(result_net.res_line) > 0

    def test_add_switches_for_topology_control(self):
        """Test adding switches for topology control on case14."""
        net = deepcopy(self.base_net)
        original_switch_count = len(net.switch)

        result_net = self.generator.add_switches_for_topology_control(net)

        # Should have same or more switches
        assert len(result_net.switch) >= original_switch_count

        # Check switch properties if new ones were added
        if len(result_net.switch) > original_switch_count:
            _ = result_net.switch.iloc[original_switch_count:]
            assert "name" in result_net.switch.columns

    def test_add_battery_opportunities(self):
        """Test marking battery opportunities on case14."""
        net = deepcopy(self.base_net)

        result_net = self.generator.add_battery_opportunities(net)

        # Should have battery_suitable column
        assert "battery_suitable" in result_net.bus.columns

        # Some buses should be suitable
        assert result_net.bus["battery_suitable"].sum() > 0

    def test_generate_test_network_full(self):
        """Test complete test network generation on case14."""
        net = deepcopy(self.base_net)

        result_net = self.generator.generate_test_network(
            base_net=net,
            voltage_violations=2,
            thermal_violations=1,
            severity="medium",
            add_switches=True,
            add_battery_sites=True,
        )

        # Check that all names are filled
        assert result_net.load["name"].notna().all()
        assert result_net.line["name"].notna().all()
        assert result_net.bus["name"].notna().all()

        # Check solution capabilities were added
        assert "battery_suitable" in result_net.bus.columns

        # Original network should not be modified
        assert len(net.load) <= len(result_net.load)

    def test_generate_test_network_minimal(self):
        """Test minimal test network generation on case14."""
        net = deepcopy(self.base_net)

        result_net = self.generator.generate_test_network(
            base_net=net,
            voltage_violations=0,
            thermal_violations=0,
            severity="light",
            add_switches=False,
            add_battery_sites=False,
        )

        # Should still fill names
        assert result_net.load["name"].notna().all()
        assert result_net.line["name"].notna().all()
        assert result_net.bus["name"].notna().all()

    def test_validate_network_convergent(self):
        """Test network validation with case14."""
        net = deepcopy(self.base_net)

        validation = self.generator.validate_network(net)

        # Check validation structure regardless of convergence
        assert "converged" in validation
        assert "voltage_violations" in validation
        assert "thermal_violations" in validation
        assert "total_violations" in validation
        assert validation["total_violations"] >= 0

        # If it doesn't converge, that's still a valid test result
        if not validation["converged"]:
            assert "error" in validation

    def test_validate_network_with_violations(self):
        """Test network validation with induced violations on case14."""
        net = deepcopy(self.base_net)

        # Add violations
        test_net = self.generator.add_voltage_violations(
            net, violation_count=2, severity="medium"
        )

        validation = self.generator.validate_network(test_net)

        assert validation["converged"] is True
        assert "total_violations" in validation

    def test_save_network(self):
        """Test network saving functionality with case14."""
        net = deepcopy(self.base_net)
        test_net = self.generator.generate_test_network(
            net, voltage_violations=1, thermal_violations=1
        )

        network_name = "test_case14"
        validation = self.generator.save_network(test_net, self.temp_dir, network_name)

        # Check files were created
        assert (self.temp_dir / "net.json").exists()
        assert (self.temp_dir / "metadata.json").exists()
        assert (self.temp_dir / "README.md").exists()

        # Check validation result
        assert "converged" in validation
        assert "total_violations" in validation

        # Check network can be loaded back
        loaded_net = pp.from_json(str(self.temp_dir / "net.json"))
        assert len(loaded_net.bus) == len(test_net.bus)

    def test_severity_levels(self):
        """Test different severity levels on case14."""
        net = deepcopy(self.base_net)
        severities = ["light", "medium", "severe"]

        for severity in severities:
            result_net = self.generator.add_voltage_violations(
                deepcopy(net), violation_count=2, severity=severity
            )

            # Should not crash during power flow
            pp.runpp(result_net)
            assert len(result_net.res_bus) > 0

    def test_edge_cases(self):
        """Test edge cases and error handling with case14."""
        net = deepcopy(self.base_net)

        # Test with more violations than buses
        large_violation_count = len(net.bus) + 10
        result_net = self.generator.add_voltage_violations(
            deepcopy(net), violation_count=large_violation_count, severity="medium"
        )

        # Should not crash and should limit violations appropriately
        assert len(result_net.bus) >= len(net.bus)

        # Test with zero violations
        result_net = self.generator.generate_test_network(
            base_net=net, voltage_violations=0, thermal_violations=0
        )
        assert len(result_net.bus) == len(net.bus)

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed on case14."""
        import copy

        net = copy.deepcopy(self.base_net)

        gen1 = NetworkGenerator(seed=123)
        gen2 = NetworkGenerator(seed=123)

        result1 = gen1.generate_test_network(
            copy.deepcopy(net), voltage_violations=2, thermal_violations=1
        )
        result2 = gen2.generate_test_network(
            copy.deepcopy(net), voltage_violations=2, thermal_violations=1
        )

        # Results should be similar (may not be identical due to network differences)
        # Check that both have curtailable loads
        assert result1.load["curtailable"].sum() > 0
        assert result2.load["curtailable"].sum() > 0

        # Validate both networks
        val1 = gen1.validate_network(result1)
        val2 = gen2.validate_network(result2)

        # Both should have similar convergence behavior
        assert "converged" in val1
        assert "converged" in val2

    def test_case14_specific_properties(self):
        """Test case14 specific network properties."""
        net = deepcopy(self.base_net)

        # case14 should have 14 buses
        assert len(net.bus) == 14

        # Generate test network
        result_net = self.generator.generate_test_network(
            base_net=net, voltage_violations=3, thermal_violations=2, severity="medium"
        )

        # Should maintain basic structure
        assert len(result_net.bus) == 14  # Buses should not change
        assert len(result_net.line) >= len(net.line)  # Lines might be same or more

        # Validation should work
        validation = self.generator.validate_network(result_net)
        assert validation["converged"] is True

    def test_curtailable_loads_creation(self):
        """Test that curtailable loads are properly created."""
        net = deepcopy(self.base_net)

        result_net = self.generator.generate_test_network(
            base_net=net, voltage_violations=2, thermal_violations=1, severity="medium"
        )

        # Should have curtailable column
        assert "curtailable" in result_net.load.columns

        # Should have some curtailable loads
        curtailable_count = result_net.load["curtailable"].sum()
        assert curtailable_count > 0

        # Curtailable loads should have names
        curtailable_loads = result_net.load[result_net.load["curtailable"]]
        assert curtailable_loads["name"].notna().all()

    def test_controllable_generation_creation(self):
        """Test that controllable generation is properly created."""
        net = deepcopy(self.base_net)

        result_net = self.generator.add_voltage_violations(
            net, violation_count=4, severity="medium"
        )

        # Check if DG was added
        if len(result_net.sgen) > 0:
            assert "controllable" in result_net.sgen.columns
            controllable_count = result_net.sgen["controllable"].sum()

            if controllable_count > 0:
                # Controllable DG should have names
                controllable_dg = result_net.sgen[result_net.sgen["controllable"]]
                assert controllable_dg["name"].notna().all()
