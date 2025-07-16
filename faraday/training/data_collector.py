"""
Enhanced training data collection for power grid optimization.
Supports multiple formats for open-source LLM fine-tuning.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import uuid

import pandapower as pp
from faraday.tools.pandapower import get_network_status
from faraday.schemas import State


class NetworkCompressor:
    """Compress network status into token-efficient representations."""

    @staticmethod
    def compress_network_status(net: pp.pandapowerNet, status: Dict) -> str:
        """Create a compressed, human-readable network summary."""
        total_buses = len(net.bus)
        total_lines = len(net.line) + len(net.trafo)

        # Extract violations
        voltage_violations = []
        thermal_violations = []

        for bus in status.get("bus_status", []):
            if bus["v_mag_pu"] > 1.05 or bus["v_mag_pu"] < 0.95:
                severity = (
                    "critical"
                    if bus["v_mag_pu"] > 1.1 or bus["v_mag_pu"] < 0.9
                    else "medium"
                )
                voltage_violations.append(
                    f"Bus {bus['index']}: {bus['v_mag_pu']:.2f}pu ({severity})"
                )

        for line in status.get("line_status", []):
            if line["loading_percent"] > 100:
                severity = "critical" if line["loading_percent"] > 150 else "medium"
                thermal_violations.append(
                    f"{line['name']}: {line['loading_percent']:.0f}% ({severity})"
                )

        # Extract resources
        switches = [
            sw["name"] for sw in status.get("switch_status", []) if sw.get("name")
        ]
        curtailable_loads = [
            load["name"]
            for load in status.get("load_status", [])
            if load.get("curtailable", False)
        ]

        # Build compact summary
        summary_parts = [
            f"Network: {total_buses}-bus grid with {total_lines} lines/transformers"
        ]

        if thermal_violations:
            thermal_summary = (
                f"Thermal violations ({len(thermal_violations)}): "
                + ", ".join(thermal_violations[:3])
            )
            if len(thermal_violations) > 3:
                thermal_summary += f" + {len(thermal_violations) - 3} more"
            summary_parts.append(thermal_summary)

        if voltage_violations:
            voltage_summary = (
                f"Voltage violations ({len(voltage_violations)}): "
                + ", ".join(voltage_violations[:3])
            )
            if len(voltage_violations) > 3:
                voltage_summary += f" + {len(voltage_violations) - 3} more"
            summary_parts.append(voltage_summary)

        if switches:
            summary_parts.append(f"Available switches: {', '.join(switches[:5])}")

        if curtailable_loads:
            summary_parts.append(
                f"Curtailable loads: {', '.join(curtailable_loads[:3])}"
            )

        summary_parts.append("Battery capacity: 3×1MW available")

        return "\n".join(summary_parts)


class DifficultyClassifier:
    """Classify network problems by difficulty level."""

    @staticmethod
    def classify_difficulty(
        net: pp.pandapowerNet, violations: Dict, actions: List
    ) -> str:
        """Classify problem difficulty based on network characteristics."""
        score = 0

        # Violation count and severity
        v_violations = len(violations.get("voltage", []))
        t_violations = len(violations.get("thermal", []))
        score += min(v_violations + t_violations, 10)

        # Network size complexity
        if len(net.bus) > 50:
            score += 3
        elif len(net.bus) > 20:
            score += 2
        elif len(net.bus) > 10:
            score += 1

        # Action coordination requirement
        if len(actions) > 5:
            score += 3
        elif len(actions) > 3:
            score += 2
        elif len(actions) > 1:
            score += 1

        # Violation severity
        for v in violations.get("voltage", []):
            if v.get("v_mag_pu", 1.0) > 1.1 or v.get("v_mag_pu", 1.0) < 0.9:
                score += 1

        for t in violations.get("thermal", []):
            if t.get("loading", 0) > 150:
                score += 1

        if score <= 3:
            return "easy"
        elif score <= 7:
            return "medium"
        else:
            return "hard"


class TrainingDataFormatter:
    """Format training data for different fine-tuning frameworks."""

    @staticmethod
    def format_openai_chat(
        system_prompt: str, network_summary: str, actions: List, explanation: str
    ) -> Dict:
        """Format for OpenAI ChatCompletion fine-tuning."""
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Analyze and fix violations in this power grid:\n\n{network_summary}",
                },
                {
                    "role": "assistant",
                    "content": f"I'll analyze the violations and provide an optimized solution.\n\n{explanation}",
                    "tool_calls": actions,
                },
            ]
        }

    @staticmethod
    def format_alpaca_instruction(
        network_summary: str, actions: List, explanation: str
    ) -> Dict:
        """Format for Alpaca instruction following."""
        # Clean actions for text output
        action_text = []
        for action in actions:
            name = action.get("name", "unknown")
            args = action.get("args", {})
            if name == "update_switch_status":
                action_text.append(
                    f"Close switch {args.get('switch_name')} to {args.get('closed')}"
                )
            elif name == "add_battery":
                action_text.append(
                    f"Add {args.get('max_energy_kw', 1000)}kW battery at bus {args.get('bus_index')}"
                )
            elif name == "curtail_load":
                action_text.append(
                    f"Curtail load {args.get('load_name')} by {args.get('curtail_percent', 0)}%"
                )

        output = f"Analysis: {explanation}\n\nActions:\n" + "\n".join(
            f"- {act}" for act in action_text
        )

        return {
            "instruction": "Analyze this power grid network and provide an optimized plan to resolve all violations. Prioritize actions that resolve multiple violations simultaneously.",
            "input": network_summary,
            "output": output,
        }

    @staticmethod
    def format_tool_calling(
        system_prompt: str, network_summary: str, actions: List
    ) -> Dict:
        """Format for tool calling fine-tuning."""
        # Clean tool calls
        clean_actions = []
        for action in actions:
            clean_action = {
                "name": action.get("name"),
                "arguments": {
                    k: v
                    for k, v in action.get("args", {}).items()
                    if k not in ["network_path"]
                },
            }
            clean_actions.append(clean_action)

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": network_summary},
                {"role": "assistant", "tool_calls": clean_actions},
            ]
        }

    @staticmethod
    def format_reasoning_chain(
        network_summary: str, actions: List, explanation: str
    ) -> Dict:
        """Format for chain-of-thought reasoning."""
        # Build step-by-step reasoning
        reasoning_steps = [
            "Let me analyze this power grid systematically:",
            "",
            "1. Violation Assessment:",
            "   - Identifying thermal overloads and voltage violations",
            "   - Assessing severity and impact on grid stability",
            "",
            "2. Resource Analysis:",
            "   - Available switches for network reconfiguration",
            "   - Curtailable loads for demand reduction",
            "   - Battery capacity for voltage support",
            "",
            "3. Solution Strategy:",
            "   - Prioritize switch operations (highest impact)",
            "   - Strategic battery placement for voltage support",
            "   - Load curtailment as needed",
            "",
            "4. Implementation Plan:",
        ]

        for i, action in enumerate(actions, 1):
            name = action.get("name", "unknown")
            args = action.get("args", {})
            if name == "update_switch_status":
                reasoning_steps.append(
                    f"   Step {i}: Close switch {args.get('switch_name')} to create alternate power flow path"
                )
            elif name == "add_battery":
                reasoning_steps.append(
                    f"   Step {i}: Install {args.get('max_energy_kw', 1000)}kW battery at bus {args.get('bus_index')} for voltage support"
                )
            elif name == "curtail_load":
                reasoning_steps.append(
                    f"   Step {i}: Reduce load {args.get('load_name')} by {args.get('curtail_percent', 0)}% to decrease demand"
                )

        reasoning_steps.extend(["", f"Explanation: {explanation}"])

        return {
            "instruction": "Analyze this power grid and provide a detailed reasoning chain for resolving violations.",
            "input": network_summary,
            "output": "\n".join(reasoning_steps),
        }


class EnhancedTrainingDataCollector:
    """Enhanced training data collector with multiple format support."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.compressor = NetworkCompressor()
        self.classifier = DifficultyClassifier()
        self.formatter = TrainingDataFormatter()

    def collect_training_sample(
        self,
        state: State,
        executed_actions: List[Dict],
        explanation: str,
        system_prompt: str,
    ) -> Dict:
        """Collect a comprehensive training sample."""

        # Load network and get status
        net = pp.from_json(state["network_file_path"])
        pp.runpp(net)
        status = get_network_status(net)

        # Compress network representation
        network_summary = self.compressor.compress_network_status(net, status)

        # Extract violations for difficulty classification
        violations = state.get(
            "violation_before_action", {"voltage": [], "thermal": []}
        )
        difficulty = self.classifier.classify_difficulty(
            net, violations, executed_actions
        )

        # Generate sample ID
        sample_id = str(uuid.uuid4())[:8]

        # Create metadata
        metadata = {
            "sample_id": sample_id,
            "timestamp": datetime.now().isoformat(),
            "network_size": len(net.bus),
            "violation_count": len(violations.get("voltage", []))
            + len(violations.get("thermal", [])),
            "action_count": len(executed_actions),
            "difficulty": difficulty,
            "iterations": state.get("iter", 1),
            "success": len(state.get("violation_after_action", {}).get("voltage", []))
            == 0
            and len(state.get("violation_after_action", {}).get("thermal", [])) == 0,
        }

        # Generate multiple training formats
        formats = {
            "openai_chat": self.formatter.format_openai_chat(
                system_prompt, network_summary, executed_actions, explanation
            ),
            "alpaca_instruction": self.formatter.format_alpaca_instruction(
                network_summary, executed_actions, explanation
            ),
            "tool_calling": self.formatter.format_tool_calling(
                system_prompt, network_summary, executed_actions
            ),
            "reasoning_chain": self.formatter.format_reasoning_chain(
                network_summary, executed_actions, explanation
            ),
        }

        # Complete training sample
        training_sample = {
            "metadata": metadata,
            "network_summary": network_summary,
            "raw_network_status": status,  # Keep for validation
            "formats": formats,
            "violations": violations,
            "actions": executed_actions,
            "explanation": explanation,
        }

        return training_sample

    def save_training_sample(self, training_sample: Dict) -> None:
        """Save training sample to multiple format files."""
        sample_id = training_sample["metadata"]["sample_id"]

        # Save complete sample
        with open(self.output_dir / f"sample_{sample_id}.json", "w") as f:
            json.dump(training_sample, f, indent=2)

        # Save individual formats for easy loading
        formats_dir = self.output_dir / "formats"
        formats_dir.mkdir(exist_ok=True)

        for format_name, format_data in training_sample["formats"].items():
            format_file = formats_dir / f"{format_name}.jsonl"
            with open(format_file, "a") as f:
                f.write(json.dumps(format_data) + "\n")

    def create_dataset_splits(
        self, train_ratio: float = 0.8, val_ratio: float = 0.15
    ) -> None:
        """Create train/validation/test splits for each format."""
        import random

        # Get all samples
        sample_files = list(self.output_dir.glob("sample_*.json"))
        random.shuffle(sample_files)

        # Calculate splits
        total = len(sample_files)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        splits = {
            "train": sample_files[:train_end],
            "validation": sample_files[train_end:val_end],
            "test": sample_files[val_end:],
        }

        # Create split files for each format
        for format_name in [
            "openai_chat",
            "alpaca_instruction",
            "tool_calling",
            "reasoning_chain",
        ]:
            format_dir = self.output_dir / "splits" / format_name
            format_dir.mkdir(parents=True, exist_ok=True)

            for split_name, files in splits.items():
                split_file = format_dir / f"{split_name}.jsonl"
                with open(split_file, "w") as f:
                    for sample_file in files:
                        with open(sample_file) as sf:
                            sample = json.load(sf)
                            f.write(json.dumps(sample["formats"][format_name]) + "\n")

    def generate_dataset_stats(self) -> Dict:
        """Generate statistics about the collected dataset."""
        sample_files = list(self.output_dir.glob("sample_*.json"))

        if not sample_files:
            return {"total_samples": 0}

        difficulties = {"easy": 0, "medium": 0, "hard": 0}
        network_sizes = []
        action_counts = []
        success_rate = []

        for sample_file in sample_files:
            with open(sample_file) as f:
                sample = json.load(f)
                metadata = sample["metadata"]
                difficulties[metadata["difficulty"]] += 1
                network_sizes.append(metadata["network_size"])
                action_counts.append(metadata["action_count"])
                success_rate.append(metadata["success"])

        return {
            "total_samples": len(sample_files),
            "difficulty_distribution": difficulties,
            "avg_network_size": sum(network_sizes) / len(network_sizes),
            "avg_actions_per_sample": sum(action_counts) / len(action_counts),
            "success_rate": sum(success_rate) / len(success_rate),
            "available_formats": [
                "openai_chat",
                "alpaca_instruction",
                "tool_calling",
                "reasoning_chain",
            ],
        }
