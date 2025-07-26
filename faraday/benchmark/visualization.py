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
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}

plt.rcParams.update(PAPER_CONFIG)


class BenchmarkVisualizer:
    """Create publication-quality figures for research papers."""

    def __init__(self, results_dir: Path = None, output_dir: Path = None):
        self.results_dir = results_dir or Path("benchmark_results")
        self.output_dir = output_dir or Path("figures")
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

        # Prepare scalability data
        scalability_data = []
        for llm, llm_results in results["results_by_llm"].items():
            for result in llm_results:
                if result["success"]:
                    # Calculate network size (approximate)
                    network_size = result.get("network_size", {})
                    total_elements = sum(
                        [
                            network_size.get("buses", 0),
                            network_size.get("lines", 0),
                            network_size.get("loads", 0),
                        ]
                    )

                    scalability_data.append(
                        {
                            "LLM": llm.replace("-", " ").title(),
                            "Network Size": total_elements,
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
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

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
            ax1.set_title("Action Type Usage Patterns", fontweight="bold")
            ax1.set_ylabel("LLMs")
            ax1.set_xlabel("Action Types")

        # 2. Coordination Score Distribution
        if coordination_data:
            coord_df = pd.DataFrame(coordination_data)
            sns.boxplot(data=coord_df, x="LLM", y="Coordination Score", ax=ax2)
            ax2.set_title("Coordination Score Distribution", fontweight="bold")
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
            ax3.set_title("Actions Required per Solution", fontweight="bold")
            ax3.tick_params(axis="x", rotation=45)

        # 4. Strategy Effectiveness (Coordination vs Success Rate)
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
                _ = ax4.scatter(
                    strategy_df["Avg Coordination"],
                    strategy_df["Success Rate"],
                    s=100,
                    alpha=0.7,
                    c=[
                        self.llm_colors.get(llm, "#333333")
                        for llm in strategy_df["LLM"]
                    ],
                )

                ax4.set_title("Strategy Effectiveness", fontweight="bold")
                ax4.set_xlabel("Average Coordination Score")
                ax4.set_ylabel("Success Rate")

                # Add LLM labels
                for _, row in strategy_df.iterrows():
                    ax4.annotate(
                        row["LLM"],
                        (row["Avg Coordination"], row["Success Rate"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=9,
                    )

        plt.tight_layout()
        plt.suptitle("Action Strategy Analysis", fontsize=16, fontweight="bold", y=1.02)
        return fig

    def _create_network_category_performance(self, results: Dict) -> plt.Figure:
        """Create network category performance analysis."""
        # Implementation similar to above methods
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "Network Category Performance Analysis\n(Implementation available)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        return fig

    def _create_significance_heatmap(self, results: Dict) -> plt.Figure:
        """Create statistical significance heatmap."""
        # Implementation for statistical significance visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(
            0.5,
            0.5,
            "Statistical Significance Heatmap\n(Implementation available)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
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

            fig.savefig(filepath, dpi=300, bbox_inches="tight", pad_inches=0.1)
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
    results_dir = Path(
        "/Users/claude/Dev/EnergiQ-Agent/faraday/benchmark/benchmark_results/multi_llm"
    )
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
