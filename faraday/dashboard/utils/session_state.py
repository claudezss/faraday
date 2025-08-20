"""
Session State Management

Centralized session state management for the dashboard.
"""

import streamlit as st
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandapower as pp

from faraday import WORKSPACE_NETWORKS
from faraday.agents.workflow.state import State


class SessionStateManager:
    """Centralized session state management."""

    @staticmethod
    def initialize():
        """Initialize session state variables."""
        if "dashboard_initialized" not in st.session_state:
            st.session_state.dashboard_initialized = True
            st.session_state.network = None
            st.session_state.network_file_name = None
            st.session_state.current_mode = None
            st.session_state.current_plan = []
            st.session_state.executed_actions = []
            st.session_state.workflow_state = None
            st.session_state.workflow_results = None
            st.session_state.activity_log = []
            st.session_state.viz_settings = {
                "plot_type": "Network Topology",
                "color_scheme": "Default",
                "show_labels": True,
                "interactive_mode": True,
            }
            st.session_state.session_start_time = datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def load_network(network_data: Dict, filename: str):
        """Load network data into session state."""
        try:
            net = pp.from_json_string(json.dumps(network_data))

            # Save network file
            network_file_path = WORKSPACE_NETWORKS / "uploaded_network.json"
            pp.to_json(net, str(network_file_path))

            # Update session state
            st.session_state.network = net
            st.session_state.network_file_name = filename
            st.session_state.network_file_path = str(network_file_path)

            # Initialize workflow state
            st.session_state.workflow_state = State(
                network_file_path=str(network_file_path),
                org_network_copy_file_path=str(network_file_path),
                editing_network_file_path=str(network_file_path),
                work_dir=str(WORKSPACE_NETWORKS.absolute()),
                messages=[],
            )

            # Log activity
            SessionStateManager.add_activity(
                "network_loaded", f"Loaded network: {filename}"
            )

        except Exception as e:
            st.error(f"Error loading network: {e}")
            raise e

    @staticmethod
    def has_network() -> bool:
        """Check if a network is loaded."""
        return st.session_state.get("network") is not None

    @staticmethod
    def get_network():
        """Get the current network."""
        return st.session_state.get("network")

    @staticmethod
    def get_network_file_name() -> str:
        """Get the network filename."""
        return st.session_state.get("network_file_name", "Unknown")

    @staticmethod
    def set_mode(mode: str):
        """Set the current operation mode."""
        st.session_state.current_mode = mode
        SessionStateManager.add_activity("mode_changed", f"Switched to {mode} mode")

    @staticmethod
    def get_current_mode() -> Optional[str]:
        """Get the current operation mode."""
        return st.session_state.get("current_mode")

    @staticmethod
    def get_current_plan() -> List[Dict[str, Any]]:
        """Get the current action plan."""
        return st.session_state.get("current_plan", [])

    @staticmethod
    def add_action_to_plan(action: Dict[str, Any]):
        """Add an action to the current plan."""
        if "current_plan" not in st.session_state:
            st.session_state.current_plan = []

        st.session_state.current_plan.append(action)
        SessionStateManager.add_activity(
            "action_added", f"Added {action['name']} to plan"
        )

    @staticmethod
    def clear_current_plan():
        """Clear the current action plan."""
        st.session_state.current_plan = []
        SessionStateManager.add_activity("plan_cleared", "Action plan cleared")

    @staticmethod
    def get_executed_actions() -> List[Dict[str, Any]]:
        """Get executed actions."""
        # For interactive mode, return interactive actions
        if (
            "executed_actions_interactive" in st.session_state
            and st.session_state.executed_actions_interactive
        ):
            return st.session_state.executed_actions_interactive
        # For auto mode, return workflow executed actions
        return st.session_state.get("executed_actions", [])

    @staticmethod
    def set_executed_actions(actions: List[Dict[str, Any]]):
        """Set executed actions."""
        st.session_state.executed_actions = actions
        SessionStateManager.add_activity(
            "actions_executed", f"Executed {len(actions)} actions"
        )

    # Chat functionality removed - not implemented in current dashboard UI

    @staticmethod
    def get_workflow_state():
        """Get the current workflow state."""
        return st.session_state.get("workflow_state")

    @staticmethod
    def update_workflow_result(result):
        """Update workflow results."""
        st.session_state.workflow_results = result
        st.session_state.workflow_state = result

        # Extract and store executed actions from workflow result
        if hasattr(result, "all_executed_actions"):
            executed_actions = result.iteration_results[-1].executed_actions
            SessionStateManager.set_executed_actions(executed_actions)

        # Set flag to auto-activate workflow results tab
        st.session_state.workflow_just_completed = True

        SessionStateManager.add_activity(
            "workflow_completed", "Workflow execution completed"
        )

    @staticmethod
    def has_workflow_results() -> bool:
        """Check if workflow results are available."""
        return st.session_state.get("workflow_results") is not None

    @staticmethod
    def get_workflow_results():
        """Get workflow results."""
        return st.session_state.get("workflow_results")

    @staticmethod
    def is_workflow_running() -> bool:
        """Check if workflow is currently running."""
        return st.session_state.get("workflow_running", False)

    @staticmethod
    def set_workflow_running(running: bool):
        """Set workflow running status."""
        st.session_state.workflow_running = running

    @staticmethod
    def get_workflow_status() -> Dict[str, Any]:
        """Get workflow status information."""
        return {
            "is_running": st.session_state.get("workflow_running", False),
            "progress": st.session_state.get("workflow_progress", 0),
            "current_step": st.session_state.get("workflow_current_step", "Ready"),
        }

    @staticmethod
    def get_activity_log() -> List[Dict[str, Any]]:
        """Get activity log."""
        return st.session_state.get("activity_log", [])

    @staticmethod
    def add_activity(action: str, details: str, status: str = "info"):
        """Add an activity to the log."""
        if "activity_log" not in st.session_state:
            st.session_state.activity_log = []

        activity = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "action": action,
            "details": details,
            "status": status,
        }

        st.session_state.activity_log.append(activity)

        # Keep only last 50 activities
        if len(st.session_state.activity_log) > 50:
            st.session_state.activity_log = st.session_state.activity_log[-50:]

    @staticmethod
    def clear_activity_log():
        """Clear the activity log."""
        st.session_state.activity_log = []

    @staticmethod
    def get_viz_settings() -> Dict[str, Any]:
        """Get visualization settings."""
        return st.session_state.get("viz_settings", {})

    @staticmethod
    def set_viz_settings(settings: Dict[str, Any]):
        """Set visualization settings."""
        if "viz_settings" not in st.session_state:
            st.session_state.viz_settings = {}

        st.session_state.viz_settings.update(settings)

    @staticmethod
    def get_session_info() -> Dict[str, Any]:
        """Get session information."""
        return {
            "start_time": st.session_state.get("session_start_time", "Unknown"),
            "network_loaded": SessionStateManager.has_network(),
            "network_file": SessionStateManager.get_network_file_name(),
            "current_mode": SessionStateManager.get_current_mode(),
            "actions_in_plan": len(SessionStateManager.get_current_plan()),
            "actions_executed": len(SessionStateManager.get_executed_actions()),
            "activities_logged": len(SessionStateManager.get_activity_log()),
        }

    @staticmethod
    def get_initial_network():
        """Get initial network state."""
        # Check if we have an interactive mode initial network saved
        if "initial_network_interactive" in st.session_state:
            return st.session_state.initial_network_interactive

        # For auto mode, get initial network from workflow results
        results = st.session_state.get("workflow_results")
        if results and hasattr(results, "org_network_copy_file_path"):
            try:
                import pandapower as pp

                return pp.from_json(results.org_network_copy_file_path)
            except Exception:
                pass

        # Try network_file_path as backup
        if results and hasattr(results, "network_file_path"):
            try:
                import pandapower as pp

                return pp.from_json(results.network_file_path)
            except Exception:
                pass

        # Fallback: try to load from initial file path (interactive mode)
        if "initial_network_file_path" in st.session_state:
            try:
                import pandapower as pp

                return pp.from_json(st.session_state.initial_network_file_path)
            except Exception:
                pass

        # Final fallback: return current network
        return st.session_state.get("network")

    @staticmethod
    def get_final_network():
        """Get final network state after workflow execution."""
        # For interactive mode, current network IS the final network after actions
        if (
            "executed_actions_interactive" in st.session_state
            and st.session_state.executed_actions_interactive
        ):
            return st.session_state.get("network")

        # For auto mode, get from workflow results
        results = st.session_state.get("workflow_results")
        if results and hasattr(results, "editing_network_file_path"):
            try:
                import pandapower as pp

                return pp.from_json(results.editing_network_file_path)
            except Exception:
                pass
        return st.session_state.get("network")  # Fallback

    @staticmethod
    def get_initial_network_state() -> Dict[str, Any]:
        """Get initial network state metrics."""
        try:
            # For interactive mode, use stored violations_before
            if "violations_before" in st.session_state:
                viola_before = st.session_state.violations_before
                return {
                    "voltage_violations": len(viola_before.voltage)
                    if viola_before.voltage
                    else 0,
                    "thermal_violations": len(viola_before.thermal)
                    if viola_before.thermal
                    else 0,
                    "total_violations": (
                        len(viola_before.voltage) if viola_before.voltage else 0
                    )
                    + (len(viola_before.thermal) if viola_before.thermal else 0),
                }

            # For auto mode, get from workflow results
            workflow_results = st.session_state.get("workflow_results")
            if (
                workflow_results
                and hasattr(workflow_results, "iteration_results")
                and workflow_results.iteration_results
            ):
                # Get violations from first iteration (before any actions)
                first_iteration = workflow_results.iteration_results[0]
                if hasattr(first_iteration, "viola_before"):
                    viola_before = first_iteration.viola_before
                    return {
                        "voltage_violations": len(viola_before.voltage)
                        if viola_before.voltage
                        else 0,
                        "thermal_violations": len(viola_before.thermal)
                        if viola_before.thermal
                        else 0,
                        "total_violations": len(viola_before.voltage)
                        + len(viola_before.thermal)
                        if viola_before.voltage and viola_before.thermal
                        else 0,
                    }
        except Exception:
            pass

        # Fallback to stored values or defaults
        return {
            "voltage_violations": st.session_state.get("initial_voltage_violations", 0),
            "thermal_violations": st.session_state.get("initial_thermal_violations", 0),
            "total_violations": st.session_state.get("initial_total_violations", 0),
        }

    @staticmethod
    def get_final_network_state() -> Dict[str, Any]:
        """Get final network state metrics."""
        try:
            # For interactive mode, calculate current violations
            if (
                "executed_actions_interactive" in st.session_state
                and st.session_state.executed_actions_interactive
            ):
                current_net = st.session_state.get("network")
                if current_net is not None:
                    from faraday.tools.pandapower import get_violations
                    import pandapower as pp

                    # Run power flow to get current state
                    pp.runpp(current_net)
                    current_violations = get_violations(current_net)
                    return {
                        "voltage_violations": len(current_violations.voltage)
                        if current_violations.voltage
                        else 0,
                        "thermal_violations": len(current_violations.thermal)
                        if current_violations.thermal
                        else 0,
                        "total_violations": (
                            len(current_violations.voltage)
                            if current_violations.voltage
                            else 0
                        )
                        + (
                            len(current_violations.thermal)
                            if current_violations.thermal
                            else 0
                        ),
                    }

            # For auto mode, get from workflow results
            workflow_results = st.session_state.get("workflow_results")
            if (
                workflow_results
                and hasattr(workflow_results, "iteration_results")
                and workflow_results.iteration_results
            ):
                # Get violations from last iteration (after all actions)
                last_iteration = workflow_results.iteration_results[-1]
                if hasattr(last_iteration, "viola_after"):
                    viola_after = last_iteration.viola_after
                    return {
                        "voltage_violations": len(viola_after.voltage)
                        if viola_after.voltage
                        else 0,
                        "thermal_violations": len(viola_after.thermal)
                        if viola_after.thermal
                        else 0,
                        "total_violations": len(viola_after.voltage)
                        + len(viola_after.thermal)
                        if viola_after.voltage and viola_after.thermal
                        else 0,
                    }
        except Exception:
            pass

        # Fallback to stored values or defaults
        return {
            "voltage_violations": st.session_state.get("final_voltage_violations", 0),
            "thermal_violations": st.session_state.get("final_thermal_violations", 0),
            "total_violations": st.session_state.get("final_total_violations", 0),
        }

    # Session reset functionality removed - not implemented in current dashboard UI

    # Session export/import functionality removed - not implemented in current dashboard UI
