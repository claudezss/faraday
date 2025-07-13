#!/usr/bin/env python3
"""
Script to generate test networks using the NetworkGenerator utility.
Creates test scenarios for cigre_mv, case30, and case118 networks.
"""

import argparse
import pandapower.networks as pn
from pathlib import Path
import sys

# Add the project root to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from energiq_agent.tools.network_generator import NetworkGenerator


def create_base_networks():
    """Create the three target base networks."""

    base_networks = {
        "cigre_mv": pn.create_cigre_network_mv(),
        "case30": pn.case30(),
        "case118": pn.case118(),
    }

    return base_networks


def generate_test_scenarios(output_dir: Path):
    """Generate test scenarios for the three target networks."""

    generator = NetworkGenerator(seed=42)
    base_networks = create_base_networks()

    scenarios = [
        # CIGRE MV scenarios
        {
            "name": "cigre_mv_light_voltage",
            "base": "cigre_mv",
            "voltage_violations": 2,
            "thermal_violations": 0,
            "severity": "light",
        },
        {
            "name": "cigre_mv_mixed_medium",
            "base": "cigre_mv",
            "voltage_violations": 3,
            "thermal_violations": 2,
            "severity": "medium",
        },
        {
            "name": "cigre_mv_thermal_severe",
            "base": "cigre_mv",
            "voltage_violations": 1,
            "thermal_violations": 3,
            "severity": "severe",
        },
        # Case30 scenarios
        {
            "name": "case30_voltage_medium",
            "base": "case30",
            "voltage_violations": 4,
            "thermal_violations": 1,
            "severity": "medium",
        },
        {
            "name": "case30_mixed_light",
            "base": "case30",
            "voltage_violations": 2,
            "thermal_violations": 2,
            "severity": "light",
        },
        # Case118 scenarios
        {
            "name": "case118_large_mixed",
            "base": "case118",
            "voltage_violations": 6,
            "thermal_violations": 4,
            "severity": "medium",
        },
        {
            "name": "case118_voltage_severe",
            "base": "case118",
            "voltage_violations": 8,
            "thermal_violations": 2,
            "severity": "severe",
        },
    ]

    results = {}

    for scenario in scenarios:
        print(f"\n🔧 Generating {scenario['name']}...")

        try:
            base_net = base_networks[scenario["base"]]

            test_net = generator.generate_test_network(
                base_net=base_net,
                voltage_violations=scenario["voltage_violations"],
                thermal_violations=scenario["thermal_violations"],
                severity=scenario["severity"],
            )

            # Save the network
            scenario_dir = output_dir / scenario["name"]
            validation = generator.save_network(
                test_net, scenario_dir, scenario["name"]
            )

            results[scenario["name"]] = validation

            if validation["converged"]:
                print(f"  ✅ Created with {validation['total_violations']} violations")
                print(f"     - Voltage: {len(validation['voltage_violations'])}")
                print(f"     - Thermal: {len(validation['thermal_violations'])}")
                print(
                    f"     - Solutions: {validation['curtailable_loads']} curtailable loads, "
                    f"{validation['switches']} switches"
                )
            else:
                print(
                    f"  ❌ Failed to converge: {validation.get('error', 'Unknown error')}"
                )

        except Exception as e:
            print(f"  ❌ Error generating {scenario['name']}: {e}")
            results[scenario["name"]] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate test networks for cigre_mv, case30, and case118"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("data/networks"),
        help="Output directory for generated networks",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 EnergiQ Agent Test Network Generator")
    print("🎯 Target networks: cigre_mv, case30, case118")
    print(f"📁 Output directory: {args.output_dir}")

    results = generate_test_scenarios(args.output_dir)

    # Summary
    print("\n📊 Summary:")
    successful = sum(1 for r in results.values() if r.get("converged", False))
    total = len(results)
    print(f"  ✅ Successfully generated: {successful}/{total} networks")

    total_violations = sum(
        r.get("total_violations", 0)
        for r in results.values()
        if r.get("converged", False)
    )
    print(f"  🎯 Total violations created: {total_violations}")

    print(f"\n🎉 Generation complete! Networks saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
