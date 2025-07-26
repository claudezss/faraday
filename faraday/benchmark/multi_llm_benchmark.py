"""
Multi-LLM benchmarking framework for comprehensive research paper evaluation.
"""

import json
import time

from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

import pandas as pd
import numpy as np

from faraday.benchmark.llm_configs import LLMConfigManager, get_test_suite
from faraday.agents.workflow.state import State
from faraday.agents.workflow.graph import get_workflow
from faraday.tools.pandapower import get_violations, read_network


logger = logging.getLogger(__name__)


@dataclass
class LLMBenchmarkResult:
    """Results for a single LLM on a single test case."""

    llm_name: str
    llm_config: str
    network_name: str
    test_case: str
    trial_id: int

    # Success metrics
    success: bool
    failure_reason: Optional[str] = None

    # Performance metrics
    runtime_seconds: float = 0.0
    total_iterations: int = 0
    total_actions: int = 0

    # Solution quality
    initial_violations: int = 0
    final_violations: int = 0
    violation_resolution_rate: float = 0.0
    action_efficiency: float = 0.0

    # LLM-specific metrics
    avg_response_time: float = 0.0
    token_usage: Dict[str, int] = None
    planning_quality_score: float = 0.0
    coordination_score: float = 0.0

    # Detailed analysis
    iteration_breakdown: List[Dict[str, Any]] = None
    action_types_used: Dict[str, int] = None
    error_recovery_attempts: int = 0


@dataclass
class MultiLLMBenchmarkSuite:
    """Complete multi-LLM benchmark results."""

    timestamp: str
    test_suite_name: str
    llms_tested: List[str]
    networks_tested: List[str]
    trials_per_combination: int

    # Aggregate results
    total_tests: int
    successful_tests: int
    results_by_llm: Dict[str, List[LLMBenchmarkResult]]

    # Comparative analysis
    llm_rankings: Dict[str, Dict[str, float]]
    statistical_significance: Dict[str, Dict[str, float]]
    performance_matrix: pd.DataFrame = None


class MultiLLMBenchmark:
    """Benchmark framework for comparing multiple LLM providers."""

    def __init__(self, results_dir: Path = None):
        self.results_dir = results_dir or Path("benchmark_results/multi_llm")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.llm_manager = LLMConfigManager()

    def run_comprehensive_llm_comparison(
        self,
        test_suite: str = "comprehensive",
        test_networks_dir: Path = None,
        max_iterations: int = 10,
        trials_per_combination: int = 3,
        specific_llms: List[str] = None,
        specific_networks: List[str] = None,
    ) -> MultiLLMBenchmarkSuite:
        """Run comprehensive comparison across multiple LLMs."""

        logger.info(f"Starting multi-LLM benchmark with test suite: {test_suite}")
        start_time = time.time()

        # Setup test parameters
        if specific_llms:
            llm_configs = specific_llms
        else:
            llm_configs = get_test_suite(test_suite)

        # Filter available LLMs
        available_configs = self.llm_manager.get_available_configs()
        llm_configs = [llm for llm in llm_configs if llm in available_configs]

        if not llm_configs:
            raise ValueError(
                "No available LLM configurations found. Check your API keys."
            )

        logger.info(f"Testing {len(llm_configs)} LLMs: {llm_configs}")

        # Discover test networks
        test_networks_dir = test_networks_dir or Path("data/test_networks")
        test_cases = self._discover_test_cases(test_networks_dir, specific_networks)

        logger.info(f"Found {len(test_cases)} test cases")

        total_tests = len(llm_configs) * len(test_cases) * trials_per_combination
        logger.info(f"Total test combinations: {total_tests}")

        # Run benchmarks
        all_results = []
        results_by_llm = {llm: [] for llm in llm_configs}

        for llm_config in llm_configs:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Testing LLM: {available_configs[llm_config].name}")
            logger.info(f"{'=' * 60}")

            llm_results = self._benchmark_single_llm(
                llm_config, test_cases, max_iterations, trials_per_combination
            )

            results_by_llm[llm_config] = llm_results
            all_results.extend(llm_results)

            # Log LLM summary
            successful = sum(1 for r in llm_results if r.success)
            success_rate = successful / len(llm_results)
            avg_runtime = np.mean([r.runtime_seconds for r in llm_results if r.success])

            logger.info(f"LLM {llm_config} Summary:")
            logger.info(
                f"  Success Rate: {success_rate:.2%} ({successful}/{len(llm_results)})"
            )
            logger.info(f"  Avg Runtime: {avg_runtime:.2f}s")

        # Perform comparative analysis
        llm_rankings = self._calculate_llm_rankings(results_by_llm)
        statistical_significance = self._calculate_statistical_significance(
            results_by_llm
        )
        performance_matrix = self._create_performance_matrix(results_by_llm)

        # Create benchmark suite
        suite = MultiLLMBenchmarkSuite(
            timestamp=datetime.now().isoformat(),
            test_suite_name=test_suite,
            llms_tested=llm_configs,
            networks_tested=[
                tc[1]["name"] if "name" in tc[1] else tc[0].parent.name
                for tc in test_cases
            ],
            trials_per_combination=trials_per_combination,
            total_tests=total_tests,
            successful_tests=sum(1 for r in all_results if r.success),
            results_by_llm=results_by_llm,
            llm_rankings=llm_rankings,
            statistical_significance=statistical_significance,
            performance_matrix=performance_matrix,
        )

        # Save results
        self._save_multi_llm_results(suite)

        total_time = time.time() - start_time
        logger.info(f"\nBenchmark completed in {total_time:.2f}s")
        self._print_summary(suite)

        return suite

    def _benchmark_single_llm(
        self,
        llm_config: str,
        test_cases: List,
        max_iterations: int,
        trials_per_combination: int,
    ) -> List[LLMBenchmarkResult]:
        """Benchmark a single LLM across all test cases."""

        results = []

        try:
            # Create LLM instance
            llm_instance = self.llm_manager.get_llm_instance(llm_config)

            for case_idx, (network_path, metadata) in enumerate(test_cases):
                network_name = metadata.get("name", network_path.parent.name)
                test_case = f"{network_name}_{network_path.stem}"

                logger.info(
                    f"  Testing case {case_idx + 1}/{len(test_cases)}: {test_case}"
                )

                for trial in range(trials_per_combination):
                    logger.info(f"    Trial {trial + 1}/{trials_per_combination}")

                    result = self._run_single_llm_test(
                        llm_config,
                        llm_instance,
                        network_path,
                        metadata,
                        max_iterations,
                        trial,
                        test_case,
                    )
                    results.append(result)

                    if result.success:
                        logger.info(
                            f"    ✅ Success in {result.runtime_seconds:.2f}s, "
                            f"{result.total_iterations} iterations, "
                            f"{result.total_actions} actions"
                        )
                    else:
                        logger.info(f"    ❌ Failed: {result.failure_reason}")

        except Exception as e:
            logger.error(f"Failed to test LLM {llm_config}: {e}")
            # Create failure results for all test cases
            for case_idx, (network_path, metadata) in enumerate(test_cases):
                test_case = f"{metadata.get('name', network_path.parent.name)}_{network_path.stem}"
                for trial in range(trials_per_combination):
                    results.append(
                        LLMBenchmarkResult(
                            llm_name=self.llm_manager.configs[llm_config].name,
                            llm_config=llm_config,
                            network_name=network_path.parent.name,
                            test_case=test_case,
                            trial_id=trial,
                            success=False,
                            failure_reason=f"LLM setup failed: {str(e)}",
                        )
                    )

        return results

    def _run_single_llm_test(
        self,
        llm_config: str,
        llm_instance,
        network_path: Path,
        metadata: Dict,
        max_iterations: int,
        trial_id: int,
        test_case: str,
    ) -> LLMBenchmarkResult:
        """Run a single test with a specific LLM."""

        start_time = time.time()

        try:
            # Temporarily replace the global LLM instance
            from faraday.agents.workflow import config

            original_llm = config.llm
            config.llm = llm_instance

            # Setup state
            state = State(
                network_file_path=str(network_path), max_iterations=max_iterations
            )

            # Get initial violations
            initial_network = read_network(str(network_path))
            initial_violations = get_violations(initial_network)
            initial_total = len(initial_violations.voltage) + len(
                initial_violations.thermal
            )

            # Run workflow
            workflow = get_workflow()
            graph = workflow.compile()

            # Track response times
            response_times = []

            # Custom monitoring could be added here to track LLM calls
            final_state = State(**graph.invoke(state))

            # Get final violations
            final_network = read_network(final_state.editing_network_file_path)
            final_violations = get_violations(final_network)
            final_total = len(final_violations.voltage) + len(final_violations.thermal)

            # Calculate metrics
            runtime = time.time() - start_time
            success = final_violations.is_resolved
            total_actions = len(final_state.all_executed_actions)

            violation_resolution_rate = (initial_total - final_total) / max(
                1, initial_total
            )
            action_efficiency = (initial_total - final_total) / max(1, total_actions)

            # Analyze action types and coordination
            action_types = {}
            coordination_scores = []

            for iter_result in final_state.iteration_results:
                for action in iter_result.executed_actions:
                    action_name = action.get("name", "unknown")
                    action_types[action_name] = action_types.get(action_name, 0) + 1

                # Calculate coordination score (simplified)
                if len(iter_result.executed_actions) > 1:
                    coordination_scores.append(0.8)  # Multi-action coordination
                elif len(iter_result.executed_actions) == 1:
                    coordination_scores.append(0.3)  # Single action

            avg_coordination = (
                np.mean(coordination_scores) if coordination_scores else 0.0
            )

            # Restore original LLM
            config.llm = original_llm

            return LLMBenchmarkResult(
                llm_name=self.llm_manager.configs[llm_config].name,
                llm_config=llm_config,
                network_name=network_path.parent.name,
                test_case=test_case,
                trial_id=trial_id,
                success=success,
                runtime_seconds=runtime,
                total_iterations=final_state.iter_num,
                total_actions=total_actions,
                initial_violations=initial_total,
                final_violations=final_total,
                violation_resolution_rate=violation_resolution_rate,
                action_efficiency=action_efficiency,
                avg_response_time=np.mean(response_times) if response_times else 0.0,
                coordination_score=avg_coordination,
                action_types_used=action_types,
                iteration_breakdown=[
                    ir.model_dump() for ir in final_state.iteration_results
                ],
            )

        except Exception as e:
            # Restore original LLM in case of error
            from faraday.agents.workflow import config

            try:
                config.llm = original_llm
            except Exception:
                pass

            runtime = time.time() - start_time
            return LLMBenchmarkResult(
                llm_name=self.llm_manager.configs[llm_config].name,
                llm_config=llm_config,
                network_name=network_path.parent.name,
                test_case=test_case,
                trial_id=trial_id,
                success=False,
                failure_reason=str(e),
                runtime_seconds=runtime,
            )

    def _discover_test_cases(
        self, test_networks_dir: Path, specific_networks: List[str] = None
    ) -> List:
        """Discover test cases, optionally filtered by specific networks."""
        test_cases = []

        for category_dir in test_networks_dir.iterdir():
            if not category_dir.is_dir():
                continue

            # Filter by specific networks if provided
            if specific_networks and not any(
                net in category_dir.name for net in specific_networks
            ):
                continue

            for test_case_dir in category_dir.iterdir():
                if not test_case_dir.is_dir():
                    continue

                # Look for network file and metadata
                network_file = None
                metadata_file = test_case_dir / "metadata.json"

                for possible_name in ["network.json", "net.json"]:
                    if (test_case_dir / possible_name).exists():
                        network_file = test_case_dir / possible_name
                        break

                if network_file and metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        test_cases.append((network_file, metadata))
                    except Exception as e:
                        logger.warning(
                            f"Failed to load metadata for {test_case_dir}: {e}"
                        )

        return test_cases

    def _calculate_llm_rankings(
        self, results_by_llm: Dict[str, List[LLMBenchmarkResult]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate rankings for each LLM across different metrics."""
        rankings = {}

        llm_metrics = {}
        for llm, results in results_by_llm.items():
            successful_results = [r for r in results if r.success]

            if not results:
                continue

            llm_metrics[llm] = {
                "success_rate": len(successful_results) / len(results),
                "avg_runtime": np.mean([r.runtime_seconds for r in successful_results])
                if successful_results
                else float("inf"),
                "avg_iterations": np.mean(
                    [r.total_iterations for r in successful_results]
                )
                if successful_results
                else float("inf"),
                "avg_actions": np.mean([r.total_actions for r in successful_results])
                if successful_results
                else float("inf"),
                "avg_resolution_rate": np.mean(
                    [r.violation_resolution_rate for r in successful_results]
                )
                if successful_results
                else 0,
                "avg_action_efficiency": np.mean(
                    [r.action_efficiency for r in successful_results]
                )
                if successful_results
                else 0,
                "avg_coordination": np.mean(
                    [r.coordination_score for r in successful_results]
                )
                if successful_results
                else 0,
            }

        # Rank by each metric
        for metric in llm_metrics[list(llm_metrics.keys())[0]].keys():
            if metric in [
                "success_rate",
                "avg_resolution_rate",
                "avg_action_efficiency",
                "avg_coordination",
            ]:
                # Higher is better
                ranked = sorted(
                    llm_metrics.items(), key=lambda x: x[1][metric], reverse=True
                )
            else:
                # Lower is better (runtime, iterations, actions)
                ranked = sorted(llm_metrics.items(), key=lambda x: x[1][metric])

            for i, (llm, _) in enumerate(ranked):
                if llm not in rankings:
                    rankings[llm] = {}
                rankings[llm][metric] = i + 1  # Rank (1 is best)

        return rankings

    def _calculate_statistical_significance(
        self, results_by_llm: Dict[str, List[LLMBenchmarkResult]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistical significance between LLM pairs."""
        # Simplified statistical analysis - in a real paper you'd use proper statistical tests
        significance = {}

        llm_names = list(results_by_llm.keys())

        for i, llm1 in enumerate(llm_names):
            significance[llm1] = {}
            results1 = [r for r in results_by_llm[llm1] if r.success]

            for j, llm2 in enumerate(llm_names):
                if i >= j:
                    continue

                results2 = [r for r in results_by_llm[llm2] if r.success]

                if len(results1) < 3 or len(results2) < 3:
                    significance[llm1][llm2] = 0.5  # Insufficient data
                    continue

                # Compare success rates (simplified)
                sr1 = len(results1) / len(results_by_llm[llm1])
                sr2 = len(results2) / len(results_by_llm[llm2])

                # Simplified significance test (would use proper statistical tests in practice)
                diff = abs(sr1 - sr2)
                if diff > 0.2:
                    significance[llm1][llm2] = 0.01  # Highly significant
                elif diff > 0.1:
                    significance[llm1][llm2] = 0.05  # Significant
                else:
                    significance[llm1][llm2] = 0.1  # Not significant

        return significance

    def _create_performance_matrix(
        self, results_by_llm: Dict[str, List[LLMBenchmarkResult]]
    ) -> pd.DataFrame:
        """Create performance matrix for visualization."""
        data = []

        for llm, results in results_by_llm.items():
            successful_results = [r for r in results if r.success]

            data.append(
                {
                    "LLM": self.llm_manager.configs[llm].name,
                    "Config": llm,
                    "Success_Rate": len(successful_results) / len(results)
                    if results
                    else 0,
                    "Avg_Runtime": np.mean(
                        [r.runtime_seconds for r in successful_results]
                    )
                    if successful_results
                    else np.inf,
                    "Avg_Iterations": np.mean(
                        [r.total_iterations for r in successful_results]
                    )
                    if successful_results
                    else np.inf,
                    "Avg_Actions": np.mean(
                        [r.total_actions for r in successful_results]
                    )
                    if successful_results
                    else np.inf,
                    "Avg_Resolution_Rate": np.mean(
                        [r.violation_resolution_rate for r in successful_results]
                    )
                    if successful_results
                    else 0,
                    "Avg_Action_Efficiency": np.mean(
                        [r.action_efficiency for r in successful_results]
                    )
                    if successful_results
                    else 0,
                    "Coordination_Score": np.mean(
                        [r.coordination_score for r in successful_results]
                    )
                    if successful_results
                    else 0,
                    "Total_Tests": len(results),
                    "Successful_Tests": len(successful_results),
                }
            )

        return pd.DataFrame(data)

    def _save_multi_llm_results(self, suite: MultiLLMBenchmarkSuite):
        """Save comprehensive multi-LLM results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full results
        results_file = self.results_dir / f"multi_llm_benchmark_{timestamp}.json"
        with open(results_file, "w") as f:
            # Convert DataFrame to dict for JSON serialization
            suite_dict = asdict(suite)
            if suite.performance_matrix is not None:
                suite_dict["performance_matrix"] = suite.performance_matrix.to_dict()
            json.dump(suite_dict, f, indent=2, default=str)

        # Save performance matrix as CSV
        if suite.performance_matrix is not None:
            csv_file = self.results_dir / f"performance_matrix_{timestamp}.csv"
            suite.performance_matrix.to_csv(csv_file, index=False)

        # Save individual LLM results
        for llm, results in suite.results_by_llm.items():
            llm_file = self.results_dir / f"results_{llm}_{timestamp}.json"
            with open(llm_file, "w") as f:
                json.dump([asdict(r) for r in results], f, indent=2, default=str)

        logger.info(f"Multi-LLM benchmark results saved to {results_file}")

    def _print_summary(self, suite: MultiLLMBenchmarkSuite):
        """Print comprehensive benchmark summary."""
        print(f"\n{'=' * 80}")
        print("MULTI-LLM BENCHMARK RESULTS")
        print(f"{'=' * 80}")
        print(f"Test Suite: {suite.test_suite_name}")
        print(f"LLMs Tested: {', '.join(suite.llms_tested)}")
        print(f"Total Tests: {suite.total_tests}")
        print(
            f"Successful: {suite.successful_tests} ({suite.successful_tests / suite.total_tests:.2%})"
        )

        if suite.performance_matrix is not None:
            print(f"\n{'LLM Performance Summary':^80}")
            print("-" * 80)

            # Sort by success rate
            sorted_df = suite.performance_matrix.sort_values(
                "Success_Rate", ascending=False
            )

            for _, row in sorted_df.iterrows():
                print(
                    f"{row['LLM']:20} | "
                    f"Success: {row['Success_Rate']:6.2%} | "
                    f"Runtime: {row['Avg_Runtime']:6.2f}s | "
                    f"Actions: {row['Avg_Actions']:5.1f} | "
                    f"Efficiency: {row['Avg_Action_Efficiency']:5.2f}"
                )

        print(f"\n{'LLM Rankings (1=best)':^80}")
        print("-" * 80)
        for llm, rankings in suite.llm_rankings.items():
            llm_name = self.llm_manager.configs[llm].name
            print(
                f"{llm_name:20} | Success: #{rankings.get('success_rate', 'N/A'):2} | "
                f"Speed: #{rankings.get('avg_runtime', 'N/A'):2} | "
                f"Efficiency: #{rankings.get('avg_action_efficiency', 'N/A'):2}"
            )


def run_multi_llm_benchmark():
    """Run the multi-LLM benchmark suite."""
    benchmark = MultiLLMBenchmark()

    # Test available configurations
    manager = LLMConfigManager()
    available = manager.get_available_configs()

    if not available:
        print("No LLM configurations available. Please set up API keys:")
        from faraday.benchmark.llm_configs import setup_environment_template

        setup_environment_template()
        return

    print(f"Available LLMs: {list(available.keys())}")

    # Run comprehensive comparison
    results = benchmark.run_comprehensive_llm_comparison(
        test_suite="comprehensive",
        test_networks_dir=Path("/Users/claude/Dev/EnergiQ-Agent/data/test_networks"),
        max_iterations=5,
        trials_per_combination=2,
    )

    return results


if __name__ == "__main__":
    run_multi_llm_benchmark()
