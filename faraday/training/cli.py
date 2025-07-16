#!/usr/bin/env python3
"""
CLI tool for managing enhanced training data for faraday.
"""

import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import shutil

from faraday.training.data_collector import EnhancedTrainingDataCollector


console = Console()


def show_dataset_stats(training_dir: Path):
    """Display comprehensive dataset statistics."""
    collector = EnhancedTrainingDataCollector(training_dir)
    stats = collector.generate_dataset_stats()

    if stats["total_samples"] == 0:
        console.print("❌ No training samples found", style="red")
        return

    console.print(f"📊 Dataset Statistics for: {training_dir}", style="cyan")
    console.print(f"Total samples: {stats['total_samples']}")
    console.print(f"Average network size: {stats['avg_network_size']:.1f} buses")
    console.print(f"Average actions per sample: {stats['avg_actions_per_sample']:.1f}")
    console.print(f"Success rate: {stats['success_rate']:.1%}")

    # Difficulty distribution table
    table = Table(title="Difficulty Distribution")
    table.add_column("Difficulty", justify="left")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    total = stats["total_samples"]
    for difficulty, count in stats["difficulty_distribution"].items():
        percentage = (count / total) * 100 if total > 0 else 0
        table.add_row(difficulty.title(), str(count), f"{percentage:.1f}%")

    console.print(table)

    # Available formats
    console.print(f"\n📝 Available formats: {', '.join(stats['available_formats'])}")


def create_splits(training_dir: Path, train_ratio: float, val_ratio: float):
    """Create train/validation/test splits."""
    collector = EnhancedTrainingDataCollector(training_dir)

    console.print(f"📂 Creating dataset splits in {training_dir}/splits/", style="cyan")
    console.print(
        f"Train: {train_ratio:.1%}, Validation: {val_ratio:.1%}, Test: {1 - train_ratio - val_ratio:.1%}"
    )

    with Progress() as progress:
        task = progress.add_task("Creating splits...", total=1)
        collector.create_dataset_splits(train_ratio, val_ratio)
        progress.update(task, completed=1)

    console.print("✅ Splits created successfully", style="green")


def export_format(training_dir: Path, format_name: str, output_file: Path):
    """Export specific format to a single file."""
    formats_dir = training_dir / "formats"
    format_file = formats_dir / f"{format_name}.jsonl"

    if not format_file.exists():
        console.print(f"❌ Format file not found: {format_file}", style="red")
        return

    console.print(f"📤 Exporting {format_name} to {output_file}", style="cyan")

    # Copy with progress
    with Progress() as progress:
        task = progress.add_task("Copying file...", total=1)
        shutil.copy2(format_file, output_file)
        progress.update(task, completed=1)

    # Count lines
    with open(output_file) as f:
        line_count = sum(1 for _ in f)

    console.print(f"✅ Exported {line_count} samples", style="green")


def validate_dataset(training_dir: Path):
    """Validate dataset integrity and quality."""
    console.print(f"🔍 Validating dataset in {training_dir}", style="cyan")

    sample_files = list(training_dir.glob("sample_*.json"))
    if not sample_files:
        console.print("❌ No sample files found", style="red")
        return

    issues = []
    valid_samples = 0

    with Progress() as progress:
        task = progress.add_task("Validating samples...", total=len(sample_files))

        for sample_file in sample_files:
            try:
                with open(sample_file) as f:
                    sample = json.load(f)

                # Validate structure
                required_keys = ["metadata", "formats", "actions", "explanation"]
                missing_keys = [key for key in required_keys if key not in sample]
                if missing_keys:
                    issues.append(f"{sample_file.name}: Missing keys {missing_keys}")
                    continue

                # Validate formats
                expected_formats = [
                    "openai_chat",
                    "alpaca_instruction",
                    "tool_calling",
                    "reasoning_chain",
                ]
                missing_formats = [
                    fmt for fmt in expected_formats if fmt not in sample["formats"]
                ]
                if missing_formats:
                    issues.append(
                        f"{sample_file.name}: Missing formats {missing_formats}"
                    )

                # Validate actions
                if not sample["actions"]:
                    issues.append(f"{sample_file.name}: No actions recorded")

                # Check for valid action structure
                for action in sample["actions"]:
                    if "name" not in action or "args" not in action:
                        issues.append(f"{sample_file.name}: Invalid action structure")
                        break

                valid_samples += 1

            except json.JSONDecodeError:
                issues.append(f"{sample_file.name}: Invalid JSON")
            except Exception as e:
                issues.append(f"{sample_file.name}: {str(e)}")

            progress.advance(task)

    # Report results
    console.print("\n📋 Validation Results:", style="cyan")
    console.print(f"Valid samples: {valid_samples}/{len(sample_files)}")

    if issues:
        console.print(f"\n⚠️  Issues found ({len(issues)}):", style="yellow")
        for issue in issues[:10]:  # Show first 10 issues
            console.print(f"  - {issue}")
        if len(issues) > 10:
            console.print(f"  ... and {len(issues) - 10} more issues")
    else:
        console.print("✅ No issues found", style="green")


def compress_legacy_data(legacy_file: Path, training_dir: Path):
    """Convert legacy training data to enhanced format."""
    if not legacy_file.exists():
        console.print(f"❌ Legacy file not found: {legacy_file}", style="red")
        return

    console.print(f"🔄 Converting legacy data from {legacy_file}", style="cyan")

    with open(legacy_file) as f:
        legacy_data = json.load(f)

    collector = EnhancedTrainingDataCollector(training_dir)
    converted = 0

    with Progress() as progress:
        task = progress.add_task("Converting samples...", total=len(legacy_data))

        for item in legacy_data:
            try:
                # Extract actions and explanation
                actions = item.get("assistant_response", {}).get("actions", [])
                explanation = item.get("assistant_response", {}).get("explanation", "")
                _ = item.get("system_prompt", "")

                if actions:  # Only convert if there are actions
                    # Create a simplified training sample
                    from datetime import datetime
                    import uuid

                    sample_id = str(uuid.uuid4())[:8]
                    network_summary = "Legacy data - network details not available"

                    training_sample = {
                        "metadata": {
                            "sample_id": sample_id,
                            "timestamp": datetime.now().isoformat(),
                            "network_size": 14,  # Assume based on legacy data
                            "violation_count": 10,  # Estimate
                            "action_count": len(actions),
                            "difficulty": "medium",
                            "iterations": 1,
                            "success": True,
                            "source": "legacy_conversion",
                        },
                        "network_summary": network_summary,
                        "formats": {
                            "alpaca_instruction": {
                                "instruction": "Analyze this power grid and resolve violations",
                                "input": "Legacy power grid data",
                                "output": explanation,
                            }
                        },
                        "actions": actions,
                        "explanation": explanation,
                    }

                    collector.save_training_sample(training_sample)
                    converted += 1

            except Exception as e:
                console.print(f"⚠️  Error converting sample: {e}", style="yellow")

            progress.advance(task)

    console.print(f"✅ Converted {converted} samples", style="green")


def main():
    parser = argparse.ArgumentParser(
        description="Manage enhanced training data for faraday",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show dataset statistics
  python -m faraday.training.cli stats workspace/training_data_enhanced
  
  # Create train/validation/test splits
  python -m faraday.training.cli splits workspace/training_data_enhanced
  
  # Export specific format
  python -m faraday.training.cli export workspace/training_data_enhanced alpaca_instruction output.jsonl
  
  # Validate dataset
  python -m faraday.training.cli validate workspace/training_data_enhanced
  
  # Convert legacy data
  python -m faraday.training.cli convert workspace/training_data.json workspace/training_data_enhanced
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument("training_dir", type=Path, help="Training data directory")

    # Splits command
    splits_parser = subparsers.add_parser("splits", help="Create dataset splits")
    splits_parser.add_argument(
        "training_dir", type=Path, help="Training data directory"
    )
    splits_parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Training split ratio"
    )
    splits_parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation split ratio"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export specific format")
    export_parser.add_argument(
        "training_dir", type=Path, help="Training data directory"
    )
    export_parser.add_argument(
        "format",
        choices=[
            "openai_chat",
            "alpaca_instruction",
            "tool_calling",
            "reasoning_chain",
        ],
    )
    export_parser.add_argument("output", type=Path, help="Output file")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument(
        "training_dir", type=Path, help="Training data directory"
    )

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert legacy training data"
    )
    convert_parser.add_argument(
        "legacy_file", type=Path, help="Legacy training data JSON file"
    )
    convert_parser.add_argument(
        "training_dir", type=Path, help="Output training data directory"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "stats":
        show_dataset_stats(args.training_dir)
    elif args.command == "splits":
        create_splits(args.training_dir, args.train_ratio, args.val_ratio)
    elif args.command == "export":
        export_format(args.training_dir, args.format, args.output)
    elif args.command == "validate":
        validate_dataset(args.training_dir)
    elif args.command == "convert":
        compress_legacy_data(args.legacy_file, args.training_dir)


if __name__ == "__main__":
    main()
