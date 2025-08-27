"""
Visualization utilities for research paper figures and analysis.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import pandapower as pp

from matplotlib.gridspec import GridSpec

# Set publication-quality style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


# Publication settings
PAPER_CONFIG = {
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.dpi": 700,
    "savefig.dpi": 700,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}

plt.rcParams.update(PAPER_CONFIG)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["figure.titlesize"] = 16


class BenchmarkVisualizer:
    """Create publication-quality figures for research papers."""

    def __init__(
        self,
        results_dir: Path = None,
        output_dir: Path = None,
        test_network_dir: Path = None,
    ):
        self.results_dir = results_dir or Path("benchmark_results")
        self.output_dir = output_dir or Path("figures")
        self.test_network_dir = test_network_dir or Path("data/test_networks")
        self.output_dir.mkdir(exist_ok=True)

        # Color schemes for different chart types
        self.llm_colors = {
            "gpt-4.1": "#1f77b4",
            "gpt-4.1-mini": "#ff7f0e",
            "claude-sonnet-4": "#9467bd",
            "claude-3-7-sonnet": "#8c564b",
            "gemini-2.5-pro": "#e377c2",
            "gemini-2.5-flash": "#7f7f7f",
        }

        # Cache for network information
        self._network_cache = {}

    def _get_network_info(self, network_name: str) -> Dict:
        """Load network information from pandapower file."""
        if network_name in self._network_cache:
            return self._network_cache[network_name]

        # Try to find the network file
        network_file = None

        # Check different network file patterns
        for subdir in ["case30", "ieee69", "cigre_mv"]:
            # Try both net.json and network.json
            for filename in ["net.json", "network.json"]:
                potential_path = (
                    self.test_network_dir / subdir / network_name / filename
                )
                if potential_path.exists():
                    network_file = potential_path
                    break
            if network_file:
                break

        if not network_file:
            # Return empty info if file not found
            self._network_cache[network_name] = {}
            return {}

        try:
            # Load network using pandapower
            net = pp.from_json(str(network_file))

            network_info = {
                "buses": len(net.bus),
                "lines": len(net.line),
                "loads": len(net.load),
                "generators": len(net.gen)
                if hasattr(net, "gen") and not net.gen.empty
                else 0,
                "transformers": len(net.trafo)
                if hasattr(net, "trafo") and not net.trafo.empty
                else 0,
                "total_elements": len(net.bus) + len(net.line) + len(net.load),
            }

            self._network_cache[network_name] = network_info
            return network_info

        except Exception as e:
            print(f"Warning: Could not load network {network_name}: {e}")
            self._network_cache[network_name] = {}
            return {}

    def create_llm_comparison_suite(
        self, multi_llm_results: Dict
    ) -> Dict[str, plt.Figure]:
        """Create comprehensive LLM comparison figure suite."""
        figures = {}

        # Load results if path provided
        if isinstance(multi_llm_results, (str, Path)):
            with open(multi_llm_results) as f:
                multi_llm_results = json.load(f)

        # Create performance matrix
        figures["performance_overview"] = self._create_performance_overview(
            multi_llm_results
        )

        # Success rate comparison
        figures["success_rates"] = self._create_success_rate_comparison(
            multi_llm_results
        )

        # Runtime and efficiency analysis
        figures["runtime_efficiency"] = self._create_runtime_efficiency_analysis(
            multi_llm_results
        )

        # Scalability analysis
        figures["scalability"] = self._create_scalability_analysis(multi_llm_results)

        # Action strategy analysis
        figures["action_strategies"] = self._create_action_strategy_analysis(
            multi_llm_results
        )

        # Network category performance
        figures["network_performance"] = self._create_network_category_performance(
            multi_llm_results
        )

        # Statistical significance heatmap
        figures["significance"] = self._create_significance_heatmap(multi_llm_results)

        return figures

    def _create_performance_overview(self, results: Dict) -> plt.Figure:
        """Create comprehensive performance overview figure."""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Parse performance matrix
        perf_data = []
        for llm, llm_results in results["results_by_llm"].items():
            successful = [r for r in llm_results if r["success"]]
            total = len(llm_results)

            if successful:
                perf_data.append(
                    {
                        "LLM": llm.replace("-", " ").title(),
                        "Success Rate": len(successful) / total,
                        "Avg Runtime": np.mean(
                            [r["runtime_seconds"] for r in successful]
                        ),
                        "Avg Actions": np.mean(
                            [r["total_actions"] for r in successful]
                        ),
                        "Resolution Rate": np.mean(
                            [r["violation_resolution_rate"] for r in successful]
                        ),
                        "Action Efficiency": np.mean(
                            [r["action_efficiency"] for r in successful]
                        ),
                        "Coordination Score": np.mean(
                            [r["coordination_score"] for r in successful]
                        ),
                    }
                )

        df = pd.DataFrame(perf_data)
        df = df.sort_values("Success Rate", ascending=False)

        # 1. Success Rate Bar Chart
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(
            range(len(df)),
            df["Success Rate"],
            color=[self.llm_colors.get(llm, "#333333") for llm in df["LLM"]],
        )
        ax1.set_title("Success Rate by LLM", fontweight="bold")
        ax1.set_ylabel("Success Rate")
        ax1.set_ylim(0, 1.1)
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df["LLM"], rotation=45, ha="right")

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.2%}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 2. Runtime vs Success Rate Scatter
        ax2 = fig.add_subplot(gs[0, 1])
        _ = ax2.scatter(
            df["Avg Runtime"],
            df["Success Rate"],
            s=100,
            alpha=0.7,
            c=[self.llm_colors.get(llm, "#333333") for llm in df["LLM"]],
        )
        ax2.set_title("Runtime vs Success Rate", fontweight="bold")
        ax2.set_xlabel("Average Runtime (seconds)")
        ax2.set_ylabel("Success Rate")

        # Add LLM labels
        for i, row in df.iterrows():
            ax2.annotate(
                row["LLM"],
                (row["Avg Runtime"], row["Success Rate"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        # 3. Action Efficiency Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        bars = ax3.bar(
            range(len(df)),
            df["Action Efficiency"],
            color=[self.llm_colors.get(llm, "#333333") for llm in df["LLM"]],
        )
        ax3.set_title("Action Efficiency", fontweight="bold")
        ax3.set_ylabel("Violations Resolved per Action")
        ax3.set_xticks(range(len(df)))
        ax3.set_xticklabels(df["LLM"], rotation=45, ha="right")

        # 4. Radar Chart for Multiple Metrics
        ax4 = fig.add_subplot(gs[1, :], projection="polar")

        # Normalize metrics for radar chart
        metrics = [
            "Success Rate",
            "Resolution Rate",
            "Action Efficiency",
            "Coordination Score",
        ]
        normalized_df = df.copy()
        for metric in metrics:
            if metric != "Success Rate":  # Success rate already 0-1
                normalized_df[metric] = (df[metric] - df[metric].min()) / (
                    df[metric].max() - df[metric].min()
                )

        # Plot top 5 LLMs
        top_llms = normalized_df.head(5)
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for i, (_, row) in enumerate(top_llms.iterrows()):
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Complete the circle

            color = self.llm_colors.get(row["LLM"], f"C{i}")
            ax4.plot(angles, values, "o-", linewidth=2, label=row["LLM"], color=color)
            ax4.fill(angles, values, alpha=0.1, color=color)

        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics)
        ax4.set_ylim(0, 1)
        ax4.set_title("Multi-Metric Performance Comparison", fontweight="bold", pad=20)
        ax4.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        plt.suptitle("LLM Performance Overview", fontsize=16, fontweight="bold")
        return fig

    def _create_success_rate_comparison(self, results: Dict) -> plt.Figure:
        """Create detailed success rate analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Aggregate data by LLM and network category
        llm_success = {}
        network_success = {}

        for llm, llm_results in results["results_by_llm"].items():
            total = len(llm_results)
            successful = sum(1 for r in llm_results if r["success"])
            llm_success[llm] = successful / total if total > 0 else 0

            # By network category
            for result in llm_results:
                network = result["network_name"]
                if network not in network_success:
                    network_success[network] = {}
                if llm not in network_success[network]:
                    network_success[network][llm] = {"total": 0, "success": 0}

                network_success[network][llm]["total"] += 1
                if result["success"]:
                    network_success[network][llm]["success"] += 1

        # 1. Overall Success Rate Bar Chart
        llm_names = list(llm_success.keys())
        success_rates = list(llm_success.values())

        bars = ax1.bar(
            llm_names,
            success_rates,
            color=[
                self.llm_colors.get(llm.replace("-", " ").title(), "#333333")
                for llm in llm_names
            ],
        )
        ax1.set_title("Overall Success Rate by LLM", fontweight="bold")
        ax1.set_ylabel("Success Rate")
        ax1.set_ylim(0, 1.1)
        ax1.tick_params(axis="x", rotation=45)

        # Add percentage labels
        for bar, rate in zip(bars, success_rates):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.02,
                f"{rate:.1%}",
                ha="center",
                va="bottom",
            )

        # 2. Success Rate by Network Category
        network_df = []
        for network, llm_data in network_success.items():
            for llm, data in llm_data.items():
                if data["total"] > 0:
                    network_df.append(
                        {
                            "Network": network,
                            "LLM": llm,
                            "Success Rate": data["success"] / data["total"],
                        }
                    )

        if network_df:
            network_pivot = pd.DataFrame(network_df).pivot(
                index="Network", columns="LLM", values="Success Rate"
            )
            network_pivot.fillna(0, inplace=True)

            sns.heatmap(
                network_pivot,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                ax=ax2,
                cbar_kws={"label": "Success Rate"},
            )
            ax2.set_title("Success Rate by Network Category", fontweight="bold")

        # 3. Success Rate Distribution
        all_rates = []
        llm_labels = []
        for llm, llm_results in results["results_by_llm"].items():
            # Calculate success rate for each trial group
            trial_groups = {}
            for result in llm_results:
                key = f"{result['network_name']}_{result['test_case']}"
                if key not in trial_groups:
                    trial_groups[key] = []
                trial_groups[key].append(result["success"])

            group_rates = [
                sum(trials) / len(trials) for trials in trial_groups.values()
            ]
            all_rates.extend(group_rates)
            llm_labels.extend([llm] * len(group_rates))

        success_dist_df = pd.DataFrame({"LLM": llm_labels, "Success Rate": all_rates})
        sns.boxplot(data=success_dist_df, x="LLM", y="Success Rate", ax=ax3)
        ax3.set_title("Success Rate Distribution", fontweight="bold")
        ax3.tick_params(axis="x", rotation=45)

        # 4. Cumulative Success Over Time/Iterations

        for llm, llm_results in results["results_by_llm"].items():
            successful_results = [r for r in llm_results if r["success"]]
            if successful_results:
                iterations = [r["total_iterations"] for r in successful_results]
                iterations.sort()
                cumulative = np.arange(1, len(iterations) + 1) / len(successful_results)

                ax4.plot(
                    iterations,
                    cumulative,
                    label=llm,
                    linewidth=2,
                    color=self.llm_colors.get(llm.replace("-", " ").title(), "#333333"),
                )

        ax4.set_title("Cumulative Success by Iterations", fontweight="bold")
        ax4.set_xlabel("Iterations Required")
        ax4.set_ylabel("Cumulative Success Rate")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle("Success Rate Analysis", fontsize=16, fontweight="bold", y=1.02)
        return fig

    def _create_runtime_efficiency_analysis(self, results: Dict) -> plt.Figure:
        """Create runtime and efficiency analysis figure."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Prepare data
        llm_data = []
        for llm, llm_results in results["results_by_llm"].items():
            successful = [r for r in llm_results if r["success"]]
            if successful:
                llm_data.append(
                    {
                        "LLM": llm.replace("-", " ").title(),
                        "Avg Runtime": np.mean(
                            [r["runtime_seconds"] for r in successful]
                        ),
                        "Runtime Std": np.std(
                            [r["runtime_seconds"] for r in successful]
                        ),
                        "Avg Actions": np.mean(
                            [r["total_actions"] for r in successful]
                        ),
                        "Action Efficiency": np.mean(
                            [r["action_efficiency"] for r in successful]
                        ),
                        "Runtimes": [r["runtime_seconds"] for r in successful],
                        "Action Counts": [r["total_actions"] for r in successful],
                    }
                )

        df = pd.DataFrame(llm_data)

        # 1. Runtime Comparison with Error Bars
        x_pos = range(len(df))
        _ = ax1.bar(
            x_pos,
            df["Avg Runtime"],
            yerr=df["Runtime Std"],
            capsize=5,
            color=[self.llm_colors.get(llm, "#333333") for llm in df["LLM"]],
        )
        ax1.set_title("Average Runtime by LLM", fontweight="bold")
        ax1.set_ylabel("Runtime (seconds)")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(df["LLM"], rotation=45, ha="right")

        # 2. Action Efficiency vs Runtime Scatter
        _ = ax2.scatter(
            df["Avg Runtime"],
            df["Action Efficiency"],
            s=100,
            alpha=0.7,
            c=[self.llm_colors.get(llm, "#333333") for llm in df["LLM"]],
        )
        ax2.set_title("Action Efficiency vs Runtime", fontweight="bold")
        ax2.set_xlabel("Average Runtime (seconds)")
        ax2.set_ylabel("Action Efficiency (Violations/Action)")

        # Add LLM labels
        for i, row in df.iterrows():
            ax2.annotate(
                row["LLM"],
                (row["Avg Runtime"], row["Action Efficiency"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        # 3. Runtime Distribution Box Plot
        runtime_data = []
        llm_labels = []
        for _, row in df.iterrows():
            runtime_data.extend(row["Runtimes"])
            llm_labels.extend([row["LLM"]] * len(row["Runtimes"]))

        runtime_df = pd.DataFrame({"LLM": llm_labels, "Runtime": runtime_data})
        sns.boxplot(data=runtime_df, x="LLM", y="Runtime", ax=ax3)
        ax3.set_title("Runtime Distribution", fontweight="bold")
        ax3.tick_params(axis="x", rotation=45)

        # 4. Actions Used Distribution
        action_data = []
        llm_labels = []
        for _, row in df.iterrows():
            action_data.extend(row["Action Counts"])
            llm_labels.extend([row["LLM"]] * len(row["Action Counts"]))

        action_df = pd.DataFrame({"LLM": llm_labels, "Actions": action_data})
        sns.violinplot(data=action_df, x="LLM", y="Actions", ax=ax4)
        ax4.set_title("Actions Required Distribution", fontweight="bold")
        ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.suptitle(
            "Runtime and Efficiency Analysis", fontsize=16, fontweight="bold", y=1.02
        )
        return fig

    def _create_scalability_analysis(self, results: Dict) -> plt.Figure:
        """Create scalability analysis figure."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Prepare scalability data using actual network information
        scalability_data = []
        for llm, llm_results in results["results_by_llm"].items():
            for result in llm_results:
                if result["success"]:
                    # Get actual network information from pandapower files
                    network_info = self._get_network_info(result["network_name"])
                    total_elements = network_info.get("total_elements", 0)

                    # Only include if we have valid network size data
                    if total_elements > 0:
                        scalability_data.append(
                            {
                                "LLM": llm.replace("-", " ").title(),
                                "Network Size": total_elements,
                                "Buses": network_info.get("buses", 0),
                                "Lines": network_info.get("lines", 0),
                                "Loads": network_info.get("loads", 0),
                                "Runtime": result["runtime_seconds"],
                                "Iterations": result["total_iterations"],
                                "Actions": result["total_actions"],
                                "Network Name": result["network_name"],
                            }
                        )

        if not scalability_data:
            # Create empty plots with message
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(
                    0.5,
                    0.5,
                    "No scalability data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            return fig

        scale_df = pd.DataFrame(scalability_data)

        # 1. Runtime vs Network Size
        for llm in scale_df["LLM"].unique():
            llm_data = scale_df[scale_df["LLM"] == llm]
            ax1.scatter(
                llm_data["Network Size"],
                llm_data["Runtime"],
                label=llm,
                alpha=0.7,
                s=50,
                color=self.llm_colors.get(llm, "#333333"),
            )

        ax1.set_title("Runtime vs Network Size", fontweight="bold")
        ax1.set_xlabel("Total Network Elements")
        ax1.set_ylabel("Runtime (seconds)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Actions Required vs Network Size
        for llm in scale_df["LLM"].unique():
            llm_data = scale_df[scale_df["LLM"] == llm]
            ax2.scatter(
                llm_data["Network Size"],
                llm_data["Actions"],
                label=llm,
                alpha=0.7,
                s=50,
                color=self.llm_colors.get(llm, "#333333"),
            )

        ax2.set_title("Actions Required vs Network Size", fontweight="bold")
        ax2.set_xlabel("Total Network Elements")
        ax2.set_ylabel("Actions Required")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Efficiency by Network Category
        network_categories = {
            "small": scale_df[scale_df["Network Size"] < 50],
            "medium": scale_df[
                (scale_df["Network Size"] >= 50) & (scale_df["Network Size"] < 100)
            ],
            "large": scale_df[scale_df["Network Size"] >= 100],
        }

        category_efficiency = []
        for category, data in network_categories.items():
            if not data.empty:
                for llm in data["LLM"].unique():
                    llm_data = data[data["LLM"] == llm]
                    if not llm_data.empty:
                        avg_runtime = llm_data["Runtime"].mean()
                        category_efficiency.append(
                            {
                                "Category": category,
                                "LLM": llm,
                                "Avg Runtime": avg_runtime,
                            }
                        )

        if category_efficiency:
            cat_df = pd.DataFrame(category_efficiency)
            cat_pivot = cat_df.pivot(
                index="LLM", columns="Category", values="Avg Runtime"
            )

            sns.heatmap(cat_pivot, annot=True, fmt=".1f", cmap="RdYlBu_r", ax=ax3)
            ax3.set_title("Runtime by Network Category", fontweight="bold")

        # 4. Scalability Trends
        # Fit trend lines for each LLM
        for llm in scale_df["LLM"].unique():
            llm_data = scale_df[scale_df["LLM"] == llm]
            if len(llm_data) > 2:
                try:
                    # Check for sufficient data variation
                    network_sizes = llm_data["Network Size"]
                    runtimes = llm_data["Runtime"]

                    # Skip if all network sizes or runtimes are the same
                    if network_sizes.nunique() < 2 or runtimes.nunique() < 2:
                        continue

                    # Skip if there are NaN or infinite values
                    if network_sizes.isna().any() or runtimes.isna().any():
                        continue
                    if np.isinf(network_sizes).any() or np.isinf(runtimes).any():
                        continue

                    # Fit polynomial trend line with error handling
                    z = np.polyfit(network_sizes, runtimes, 1)
                    p = np.poly1d(z)

                    x_trend = np.linspace(network_sizes.min(), network_sizes.max(), 100)
                    ax4.plot(
                        x_trend,
                        p(x_trend),
                        "--",
                        color=self.llm_colors.get(llm, "#333333"),
                        label=f"{llm} trend",
                        alpha=0.8,
                    )
                except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
                    # Skip this LLM if polynomial fitting fails
                    print(f"Warning: Could not fit trend line for {llm}: {e}")
                    continue

        ax4.set_title("Runtime Scalability Trends", fontweight="bold")
        ax4.set_xlabel("Network Size")
        ax4.set_ylabel("Predicted Runtime (seconds)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle("Scalability Analysis", fontsize=16, fontweight="bold", y=1.02)
        return fig

    def _create_action_strategy_analysis(self, results: Dict) -> plt.Figure:
        """Analyze action strategies used by different LLMs."""
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 24))

        # Collect action strategy data
        action_usage = {}
        coordination_data = []

        for llm, llm_results in results["results_by_llm"].items():
            action_usage[llm] = {}

            for result in llm_results:
                if result["success"] and "action_types_used" in result:
                    # Action type usage
                    for action_type, count in result["action_types_used"].items():
                        if action_type not in action_usage[llm]:
                            action_usage[llm][action_type] = 0
                        action_usage[llm][action_type] += count

                    # Coordination score
                    coordination_data.append(
                        {
                            "LLM": llm.replace("-", " ").title(),
                            "Coordination Score": result.get("coordination_score", 0),
                            "Total Actions": result["total_actions"],
                        }
                    )

        # 1. Action Type Usage Heatmap
        all_actions = set()
        for llm_actions in action_usage.values():
            all_actions.update(llm_actions.keys())

        usage_matrix = []
        llm_names = []
        for llm, actions in action_usage.items():
            llm_names.append(llm.replace("-", " ").title())
            row = [actions.get(action, 0) for action in sorted(all_actions)]
            usage_matrix.append(row)

        if usage_matrix and all_actions:
            usage_df = pd.DataFrame(
                usage_matrix, index=llm_names, columns=sorted(all_actions)
            )

            # Normalize by row (percentage of each LLM's total actions)
            usage_normalized = usage_df.div(usage_df.sum(axis=1), axis=0)

            sns.heatmap(usage_normalized, annot=True, fmt=".2f", cmap="Blues", ax=ax1)
            ax1.set_title("(a) Action Type Usage Patterns", fontweight="bold")
            ax1.set_ylabel("LLMs")
            ax1.set_xlabel("Action Types")

        # 2. Coordination Score Distribution
        if coordination_data:
            coord_df = pd.DataFrame(coordination_data)
            sns.boxplot(data=coord_df, x="LLM", y="Coordination Score", ax=ax2)
            ax2.set_title("(b) Coordination Score Distribution", fontweight="bold")
            ax2.tick_params(axis="x", rotation=45)

        # 3. Actions per Solution
        actions_per_solution = []
        for llm, llm_results in results["results_by_llm"].items():
            successful = [r for r in llm_results if r["success"]]
            if successful:
                actions_list = [r["total_actions"] for r in successful]
                actions_per_solution.extend(
                    [
                        (llm.replace("-", " ").title(), actions)
                        for actions in actions_list
                    ]
                )

        if actions_per_solution:
            actions_df = pd.DataFrame(actions_per_solution, columns=["LLM", "Actions"])
            sns.violinplot(data=actions_df, x="LLM", y="Actions", ax=ax3)
            ax3.set_title("(c) Actions Required per Solution", fontweight="bold")
            ax3.tick_params(axis="x", rotation=45)

        # 4. Strategy Effectiveness (Coordination vs Success Rate) - IMPROVED
        if coordination_data:
            strategy_effectiveness = []
            for llm, llm_results in results["results_by_llm"].items():
                successful = [r for r in llm_results if r["success"]]
                total = len(llm_results)

                if successful:
                    avg_coordination = np.mean(
                        [r.get("coordination_score", 0) for r in successful]
                    )
                    success_rate = len(successful) / total

                    strategy_effectiveness.append(
                        {
                            "LLM": llm.replace("-", " ").title(),
                            "Avg Coordination": avg_coordination,
                            "Success Rate": success_rate,
                        }
                    )

            if strategy_effectiveness:
                strategy_df = pd.DataFrame(strategy_effectiveness)

                # Create scatter plot with larger markers
                _ = ax4.scatter(
                    strategy_df["Avg Coordination"],
                    strategy_df["Success Rate"],
                    s=200,  # Increased marker size
                    alpha=0.7,
                    c=[
                        self.llm_colors.get(llm, "#333333")
                        for llm in strategy_df["LLM"]
                    ],
                    edgecolors="black",  # Add edge for better visibility
                    linewidth=1,
                )

                ax4.set_title("(d) Strategy Effectiveness", fontweight="bold")
                ax4.set_xlabel("Average Coordination Score")
                ax4.set_ylabel("Success Rate")

                # Add grid for better readability
                ax4.grid(True, alpha=0.3)

                # Improved label positioning to avoid overlap
                from adjustText import adjust_text

                texts = []

                for _, row in strategy_df.iterrows():
                    text = ax4.annotate(
                        row["LLM"],
                        (row["Avg Coordination"], row["Success Rate"]),
                        fontsize=13,  # Increased font size
                        fontweight="bold",
                        ha="center",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            alpha=0.8,
                            edgecolor="gray",
                            linewidth=0.5,
                        ),
                    )
                    texts.append(text)

                # Use adjust_text to prevent overlapping (requires: pip install adjusttext)
                # If adjusttext is not available, use alternative positioning
                try:
                    adjust_text(
                        texts,
                        arrowprops=dict(
                            arrowstyle="->",
                            connectionstyle="arc3,rad=0.1",
                            color="gray",
                            alpha=0.7,
                            lw=1,
                        ),
                    )
                except ImportError:
                    # Alternative: manual offset positioning if adjusttext not available
                    offsets = [
                        (10, 10),
                        (-10, 10),
                        (10, -10),
                        (-10, -10),
                        (15, 0),
                        (-15, 0),
                        (0, 15),
                        (0, -15),
                    ]

                    for i, (_, row) in enumerate(strategy_df.iterrows()):
                        offset_x, offset_y = offsets[i % len(offsets)]
                        ax4.annotate(
                            row["LLM"],
                            (row["Avg Coordination"], row["Success Rate"]),
                            xytext=(offset_x, offset_y),
                            textcoords="offset points",
                            fontsize=10,
                            fontweight="bold",
                            ha="center",
                            va="center",
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="white",
                                alpha=0.8,
                                edgecolor="gray",
                                linewidth=0.5,
                            ),
                            arrowprops=dict(
                                arrowstyle="->",
                                connectionstyle="arc3,rad=0.1",
                                color="gray",
                                alpha=0.7,
                                lw=1,
                            ),
                        )

                # Set reasonable axis limits with padding
                x_margin = (
                    strategy_df["Avg Coordination"].max()
                    - strategy_df["Avg Coordination"].min()
                ) * 0.1
                y_margin = (
                    strategy_df["Success Rate"].max()
                    - strategy_df["Success Rate"].min()
                ) * 0.1

                ax4.set_xlim(
                    strategy_df["Avg Coordination"].min() - x_margin,
                    strategy_df["Avg Coordination"].max() + x_margin,
                )
                ax4.set_ylim(
                    strategy_df["Success Rate"].min() - y_margin,
                    strategy_df["Success Rate"].max() + y_margin,
                )

        # Align all subplots to the left
        plt.tight_layout()
        plt.subplots_adjust(left=0.15)  # Increase left margin to align subplots

        # Alternative: More precise alignment using subplot positions
        # Get current positions and align them
        for ax in [ax1, ax2, ax3, ax4]:
            pos = ax.get_position()
            ax.set_position([0.15, pos.y0, pos.width, pos.height])

        plt.suptitle("Action Strategy Analysis", fontsize=16, fontweight="bold", y=1.02)
        return fig

    def _create_network_category_performance(self, results: Dict) -> plt.Figure:
        """Create network category performance analysis."""
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 24))

        # Categorize networks by type and complexity
        network_categories = {
            "case30": ["case30_light", "case30_medium"],
            "ieee69": [
                "ieee69_drop_1_line",
                "ieee69_1_large_loads",
                "ieee69_2_medium_loads",
            ],
            "cigre_mv": [
                "cigre_mv_1_large_load",
                "cigre_mv_2_small_loads",
                "cigre_mv_drop_1_line",
            ],
        }

        # Collect performance data by network category
        category_data = []
        network_performance = {}

        for llm, llm_results in results["results_by_llm"].items():
            for result in llm_results:
                network_name = result["network_name"]

                # Find category
                category = "other"
                for cat, networks in network_categories.items():
                    if network_name in networks:
                        category = cat
                        break

                if category not in network_performance:
                    network_performance[category] = {}
                if llm not in network_performance[category]:
                    network_performance[category][llm] = []

                network_performance[category][llm].append(
                    {
                        "success": result["success"],
                        "runtime": result["runtime_seconds"],
                        "actions": result["total_actions"],
                        "efficiency": result["action_efficiency"],
                        "resolution_rate": result["violation_resolution_rate"],
                    }
                )

                if result["success"]:
                    # Get network info for additional analysis
                    network_info = self._get_network_info(network_name)

                    category_data.append(
                        {
                            "LLM": llm.replace("-", " ").title(),
                            "Category": category,
                            "Network": network_name,
                            "Success Rate": 1.0,
                            "Runtime": result["runtime_seconds"],
                            "Actions": result["total_actions"],
                            "Efficiency": result["action_efficiency"],
                            "Network Size": network_info.get("total_elements", 0),
                        }
                    )

        # 1. Success Rate by Network Category
        if category_data:
            cat_df = pd.DataFrame(category_data)

            # Calculate success rates by category and LLM
            success_summary = []
            for category in network_categories.keys():
                for llm, data in network_performance.get(category, {}).items():
                    if data:
                        success_rate = sum(1 for r in data if r["success"]) / len(data)
                        success_summary.append(
                            {
                                "Category": category,
                                "LLM": llm.replace("-", " ").title(),
                                "Success Rate": success_rate,
                            }
                        )

            if success_summary:
                success_df = pd.DataFrame(success_summary)
                success_pivot = success_df.pivot(
                    index="LLM", columns="Category", values="Success Rate"
                )
                success_pivot.fillna(0, inplace=True)

                sns.heatmap(
                    success_pivot,
                    annot=True,
                    fmt=".2f",
                    cmap="RdYlGn",
                    ax=ax1,
                    cbar_kws={"label": "Success Rate"},
                )
                ax1.set_title("(a) Success Rate by Network Category", fontweight="bold")

        # 2. Runtime Performance by Category
        if category_data:
            runtime_data = []
            for category, llm_data in network_performance.items():
                for llm, results_list in llm_data.items():
                    successful_results = [r for r in results_list if r["success"]]
                    if successful_results:
                        avg_runtime = np.mean(
                            [r["runtime"] for r in successful_results]
                        )
                        runtime_data.append(
                            {
                                "Category": category,
                                "LLM": llm.replace("-", " ").title(),
                                "Avg Runtime": avg_runtime,
                            }
                        )

            if runtime_data:
                runtime_df = pd.DataFrame(runtime_data)
                runtime_pivot = runtime_df.pivot(
                    index="LLM", columns="Category", values="Avg Runtime"
                )

                sns.heatmap(
                    runtime_pivot,
                    annot=True,
                    fmt=".1f",
                    cmap="RdYlBu_r",
                    ax=ax2,
                    cbar_kws={"label": "Runtime (seconds)"},
                )
                ax2.set_title(
                    "(b) Average Runtime by Network Category", fontweight="bold"
                )

        # 3. Action Efficiency by Category
        if category_data:
            valid_data = cat_df[
                cat_df["Network Size"] > 0
            ]  # Only include networks with valid size data
            if not valid_data.empty:
                sns.boxplot(
                    data=valid_data, x="Category", y="Efficiency", hue="LLM", ax=ax3
                )
                ax3.set_title(
                    "(c) Action Efficiency by Network Category", fontweight="bold"
                )
                ax3.set_ylabel("Actions per Violation Resolved")
                # Adjust legend position for vertical layout
                ax3.legend(loc="upper left", frameon=True)

        # 4. Network Complexity Analysis
        complexity_metrics = []
        for llm, llm_results in results["results_by_llm"].items():
            for result in llm_results:
                if result["success"]:
                    network_info = self._get_network_info(result["network_name"])
                    network_size = network_info.get("total_elements", 0)

                    if network_size > 0:
                        complexity_metrics.append(
                            {
                                "LLM": llm.replace("-", " ").title(),
                                "Network": result["network_name"],
                                "Network Size": network_size,
                                "Runtime": result["runtime_seconds"],
                                "Actions": result["total_actions"],
                            }
                        )

        if complexity_metrics:
            comp_df = pd.DataFrame(complexity_metrics)

            # Scatter plot of network size vs runtime
            for llm in comp_df["LLM"].unique():
                llm_data = comp_df[comp_df["LLM"] == llm]
                ax4.scatter(
                    llm_data["Network Size"],
                    llm_data["Runtime"],
                    label=llm,
                    alpha=0.7,
                    s=60,
                    color=self.llm_colors.get(llm.lower().replace(" ", "-"), "#333333"),
                )

            ax4.set_title("(d) Runtime vs Network Size", fontweight="bold")
            ax4.set_xlabel("Network Elements (Buses + Lines + Loads)")
            ax4.set_ylabel("Runtime (seconds)")
            ax4.legend(loc="best", frameon=True)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle(
            "Network Category Performance Analysis",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
        return fig

    def _create_significance_heatmap(self, results: Dict) -> plt.Figure:
        """Create statistical significance heatmap."""
        from scipy import stats

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Prepare data for statistical analysis
        llm_performance_data = {}

        for llm, llm_results in results["results_by_llm"].items():
            llm_performance_data[llm] = {
                "success_rates": [],
                "runtimes": [],
                "actions": [],
                "efficiencies": [],
            }

            # Group by network to get per-network performance
            network_groups = {}
            for result in llm_results:
                network = result["network_name"]
                if network not in network_groups:
                    network_groups[network] = []
                network_groups[network].append(result)

            # Calculate metrics per network
            for network, network_results in network_groups.items():
                success_rate = sum(1 for r in network_results if r["success"]) / len(
                    network_results
                )
                llm_performance_data[llm]["success_rates"].append(success_rate)

                successful_results = [r for r in network_results if r["success"]]
                if successful_results:
                    avg_runtime = np.mean(
                        [r["runtime_seconds"] for r in successful_results]
                    )
                    avg_actions = np.mean(
                        [r["total_actions"] for r in successful_results]
                    )
                    avg_efficiency = np.mean(
                        [r["action_efficiency"] for r in successful_results]
                    )

                    llm_performance_data[llm]["runtimes"].append(avg_runtime)
                    llm_performance_data[llm]["actions"].append(avg_actions)
                    llm_performance_data[llm]["efficiencies"].append(avg_efficiency)

        # Create significance matrices for different metrics
        llms = list(llm_performance_data.keys())
        metrics = ["success_rates", "runtimes", "actions", "efficiencies"]
        metric_names = ["Success Rate", "Runtime", "Actions", "Efficiency"]

        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = [ax1, ax2, ax3, ax4][i]

            # Create p-value matrix
            n_llms = len(llms)
            p_values = np.ones((n_llms, n_llms))

            for j, llm1 in enumerate(llms):
                for k, llm2 in enumerate(llms):
                    if (
                        j != k
                        and len(llm_performance_data[llm1][metric]) > 1
                        and len(llm_performance_data[llm2][metric]) > 1
                    ):
                        try:
                            # Perform Mann-Whitney U test (non-parametric)
                            data1 = llm_performance_data[llm1][metric]
                            data2 = llm_performance_data[llm2][metric]

                            if len(data1) > 0 and len(data2) > 0:
                                stat, p_val = stats.mannwhitneyu(
                                    data1, data2, alternative="two-sided"
                                )
                                p_values[j, k] = p_val
                        except (ValueError, RuntimeWarning):
                            p_values[j, k] = 1.0  # No significance if test fails

            # Convert p-values to significance levels
            significance_matrix = np.zeros_like(p_values)
            significance_matrix[p_values < 0.001] = 3  # ***
            significance_matrix[(p_values >= 0.001) & (p_values < 0.01)] = 2  # **
            significance_matrix[(p_values >= 0.01) & (p_values < 0.05)] = 1  # *
            significance_matrix[p_values >= 0.05] = 0  # not significant

            # Create heatmap
            llm_display_names = [llm.replace("-", " ").title() for llm in llms]

            # Create custom colormap for significance levels
            colors = [
                "#f0f0f0",
                "#ffeda0",
                "#feb24c",
                "#f03b20",
            ]  # white, light yellow, orange, red
            from matplotlib.colors import ListedColormap

            cmap = ListedColormap(colors)

            ax.imshow(significance_matrix, cmap=cmap, vmin=0, vmax=3)

            # Add significance symbols
            for j in range(n_llms):
                for k in range(n_llms):
                    if j != k:
                        sig_level = significance_matrix[j, k]
                        if sig_level == 3:
                            text = "***"
                        elif sig_level == 2:
                            text = "**"
                        elif sig_level == 1:
                            text = "*"
                        else:
                            text = "n.s."

                        # Add p-value text
                        p_val_text = (
                            f"p={p_values[j, k]:.3f}"
                            if p_values[j, k] < 0.999
                            else "p>0.05"
                        )
                        ax.text(
                            k,
                            j,
                            f"{text}\n{p_val_text}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="black" if sig_level < 2 else "white",
                        )

            ax.set_xticks(range(n_llms))
            ax.set_yticks(range(n_llms))
            ax.set_xticklabels(llm_display_names, rotation=45, ha="right")
            ax.set_yticklabels(llm_display_names)
            ax.set_title(f"{metric_name} Significance", fontweight="bold")

            # Add diagonal line (self-comparison)
            for j in range(n_llms):
                ax.text(
                    j, j, "-", ha="center", va="center", fontsize=12, fontweight="bold"
                )

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#f0f0f0", label="Not significant (p ≥ 0.05)"),
            Patch(facecolor="#ffeda0", label="* (p < 0.05)"),
            Patch(facecolor="#feb24c", label="** (p < 0.01)"),
            Patch(facecolor="#f03b20", label="*** (p < 0.001)"),
        ]
        fig.legend(
            handles=legend_elements, loc="center", bbox_to_anchor=(0.5, 0.02), ncol=4
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for legend
        plt.suptitle(
            "Statistical Significance Analysis (Mann-Whitney U Test)",
            fontsize=16,
            fontweight="bold",
            y=0.95,
        )
        return fig

    def save_all_figures(self, figures: Dict[str, plt.Figure], prefix: str = ""):
        """Save all figures to the output directory."""
        saved_files = []

        for name, fig in figures.items():
            filename = (
                f"{prefix}{name}_analysis.png" if prefix else f"{name}_analysis.png"
            )
            filepath = self.output_dir / filename

            fig.savefig(filepath, dpi=700, bbox_inches="tight", pad_inches=0.1)
            saved_files.append(filepath)

            # Also save as PDF for publications
            pdf_path = filepath.with_suffix(".pdf")
            fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.1)
            saved_files.append(pdf_path)

        print(f"Saved {len(saved_files)} figure files to {self.output_dir}")
        return saved_files

    def create_paper_ready_figures(self, results_file: Path) -> List[Path]:
        """Create complete set of paper-ready figures."""
        with open(results_file) as f:
            results = json.load(f)

        # Create all figure types
        figures = self.create_llm_comparison_suite(results)

        # Save with publication-ready settings
        saved_files = self.save_all_figures(figures, prefix="paper_")

        print(f"Generated {len(figures)} publication-ready figures")
        return saved_files


def create_sample_figures():
    """Create sample figures for demonstration."""
    visualizer = BenchmarkVisualizer()

    # Look for recent benchmark results
    results_dir = Path("benchmark_results/multi_llm")
    if results_dir.exists():
        result_files = list(results_dir.glob("multi_llm_benchmark_*.json"))
        if result_files:
            latest_results = max(result_files, key=lambda p: p.stat().st_mtime)
            print(f"Found benchmark results: {latest_results}")

            figures = visualizer.create_paper_ready_figures(latest_results)
            return figures

    print("No benchmark results found. Run benchmark first:")
    print("python -m faraday.benchmark.multi_llm_benchmark")
    return []


if __name__ == "__main__":
    create_sample_figures()
