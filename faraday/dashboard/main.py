"""
Faraday Dashboard V2 - Main Application

Advanced Streamlit dashboard for power grid violation analysis and resolution
with both automated and interactive modes.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import pandas as pd

from faraday.dashboard.components.network_viz import NetworkVisualization

from faraday.dashboard.components.comparison_view import ComparisonView
from faraday.dashboard.utils.session_state import SessionStateManager
from faraday.dashboard.utils.data_processing import NetworkDataProcessor
from faraday.agents.workflow.graph import get_workflow
from faraday.agents.workflow.state import State
from faraday.tools.pandapower import (
    get_voltage_thresholds,
    set_voltage_thresholds,
)
from faraday import DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit
st.set_page_config(
    page_title="⚡ Faraday Power Grid Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/faraday/docs",
        "Report a bug": "https://github.com/faraday/issues",
        "About": "# Faraday Dashboard v2.0\nAdvanced AI-powered power grid optimization",
    },
)


def apply_custom_styles():
    """Apply custom CSS styling."""
    st.markdown(
        """
    <style>
    /* Main dashboard styling */
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2980b9 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .status-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .violation-card {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .success-card {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .error-card {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Mode selection buttons */
    .mode-button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        color: white;
        padding: 1rem 2rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .mode-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Action plan styling */
    .action-item {
        background: #f8f9ff;
        border: 1px solid #e1e5fe;
        border-radius: 6px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    
    /* Progress indicators */
    .progress-container {
        background: #f1f3f4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Network metrics */
    .metric-container {
        display: flex;
        justify-content: space-around;
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .metric-item {
        text-align: center;
        padding: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    
    .metric-label {
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    
    /* Chat interface */
    .chat-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        min-height: 300px;
        border: 1px solid #e1e5fe;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 0.5rem 1rem;
            text-align: center;
        }
        
        .metric-container {
            flex-direction: column;
        }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def get_available_networks() -> List[Tuple[str, str]]:
    """Get list of available test networks with display names and file paths."""
    networks = []

    # Add test networks
    test_networks_dir = DATA_DIR / "test_networks"
    if test_networks_dir.exists():
        for network_dir in test_networks_dir.iterdir():
            if network_dir.is_dir():
                for scenario_dir in network_dir.iterdir():
                    if scenario_dir.is_dir():
                        network_file = scenario_dir / "network.json"
                        if network_file.exists():
                            # Create friendly display name
                            network_type = network_dir.name.upper()
                            scenario = scenario_dir.name.replace("_", " ").title()
                            display_name = f"🧪 {network_type}: {scenario}"
                            networks.append((display_name, str(network_file)))

    # Sort by display name
    networks.sort(key=lambda x: x[0])
    return networks


def load_network_from_path(network_path: str, display_name: str):
    """Load a network from a file path."""
    try:
        with open(network_path, "r") as f:
            network_data = json.load(f)

        # Extract filename for session state
        filename = Path(network_path).name
        SessionStateManager.load_network(network_data, f"{display_name} ({filename})")
        return True
    except Exception as e:
        st.error(f"❌ Error loading network: {e}")
        return False


def render_header():
    """Render the main dashboard header."""
    st.markdown(
        """
    <div class="main-header">
        <h1>⚡ Faraday Power Grid Dashboard</h1>
        <p>AI-Powered Electrical Network Violation Analysis & Resolution</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the configuration sidebar."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Network file management
        st.subheader("📁 Network Selection")

        # Get available networks
        available_networks = get_available_networks()

        if available_networks:
            # Create selection options
            network_options = ["Select a network..."] + [
                name for name, _ in available_networks
            ]

            # Use session state to track the current selection to prevent infinite reloads
            if "selected_network_name" not in st.session_state:
                st.session_state.selected_network_name = "Select a network..."

            # Determine the current index based on loaded network or default
            current_network_name = SessionStateManager.get_network_file_name()
            default_index = 0

            # If a network is already loaded, try to find its index
            if current_network_name and current_network_name != "Unknown":
                for i, option in enumerate(network_options):
                    if isinstance(option, str) and current_network_name in option:
                        default_index = i
                        break

            selected_network = st.selectbox(
                "Choose Network",
                network_options,
                index=default_index,
                help="Select a pre-configured test network from the available options",
                key="network_selector",
            )

            # Only load network if selection changed and it's not the placeholder
            if (
                selected_network != st.session_state.selected_network_name
                and selected_network != "Select a network..."
            ):
                # Find the corresponding network path
                network_path = None
                for name, path in available_networks:
                    if name == selected_network:
                        network_path = path
                        break

                if network_path:
                    # Load the selected network
                    if load_network_from_path(network_path, selected_network):
                        st.session_state.selected_network_name = selected_network
                        st.success(f"✅ Loaded: {selected_network}")
                        st.rerun()
                    else:
                        # Reset selection on failure
                        st.session_state.selected_network_name = "Select a network..."

            # Display current network info if loaded
            if SessionStateManager.has_network():
                current_name = SessionStateManager.get_network_file_name()
                st.info(f"📊 Current: {current_name}")
        else:
            st.error("❌ No test networks found in data directory")

        # Voltage thresholds configuration
        st.subheader("🔋 Voltage Thresholds")
        current_thresholds = get_voltage_thresholds()

        col1, col2 = st.columns(2)
        with col1:
            v_min = st.number_input(
                "Min (p.u.)",
                min_value=0.80,
                max_value=1.00,
                value=current_thresholds.v_min,
                step=0.01,
                format="%.2f",
            )

        with col2:
            v_max = st.number_input(
                "Max (p.u.)",
                min_value=1.00,
                max_value=1.20,
                value=current_thresholds.v_max,
                step=0.01,
                format="%.2f",
            )

        if st.button("Update Thresholds"):
            set_voltage_thresholds(v_max=v_max, v_min=v_min)
            st.success("✅ Thresholds updated!")
            st.rerun()

        # Session management
        st.subheader("📊 Session Info")
        session_info = SessionStateManager.get_session_info()
        for key, value in session_info.items():
            st.metric(key.replace("_", " ").title(), value)

        # Export functionality removed - not implemented

        # Debug info (if in debug mode)
        if st.checkbox("🐛 Debug Mode"):
            st.subheader("Debug Information")
            st.json(st.session_state.to_dict())


def render_network_status():
    """Render network status overview."""
    if not SessionStateManager.has_network():
        st.info("📁 Please select a network from the sidebar to begin analysis.")
        return

    # Get network metrics
    processor = NetworkDataProcessor()
    metrics = processor.get_network_metrics()
    violations = processor.get_violations_summary()

    # Network health overview
    st.subheader("🏥 Network Health Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Buses",
            metrics.get("total_buses", 0),
            help="Total number of buses in the network",
        )

    with col2:
        st.metric(
            "Total Lines",
            metrics.get("total_lines", 0),
            help="Total number of lines and transformers",
        )

    with col3:
        voltage_health = (
            "🟢 Healthy"
            if violations.get("voltage_violations", 0) == 0
            else f"🔴 {violations.get('voltage_violations', 0)} Violations"
        )
        st.metric("Voltage Status", voltage_health, help="Voltage violation status")

    with col4:
        thermal_health = (
            "🟢 Healthy"
            if violations.get("thermal_violations", 0) == 0
            else f"🔴 {violations.get('thermal_violations', 0)} Violations"
        )
        st.metric("Thermal Status", thermal_health, help="Thermal violation status")

    with col5:
        disconnected_health = (
            "🟢 Connected"
            if violations.get("disconnected_buses", 0) == 0
            else f"🔴 {violations.get('disconnected_buses', 0)} Disconnected"
        )
        st.metric("Connectivity", disconnected_health, help="Disconnected bus status")

    # Detailed violation information
    total_violations = violations.get("total_violations", 0)
    if total_violations > 0:
        st.markdown('<div class="violation-card">', unsafe_allow_html=True)
        violation_details = []
        if violations.get("voltage_violations", 0) > 0:
            violation_details.append(
                f"{violations.get('voltage_violations', 0)} voltage"
            )
        if violations.get("thermal_violations", 0) > 0:
            violation_details.append(
                f"{violations.get('thermal_violations', 0)} thermal"
            )
        if violations.get("disconnected_buses", 0) > 0:
            violation_details.append(
                f"{violations.get('disconnected_buses', 0)} disconnected bus"
            )

        violation_text = ", ".join(violation_details)
        st.warning(
            f"⚠️ Network has {total_violations} violations requiring attention: {violation_text}"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-card">', unsafe_allow_html=True)
        st.success("✅ Network is operating within all specified limits")
        st.markdown("</div>", unsafe_allow_html=True)


def render_mode_selection():
    """Render mode selection interface."""
    if not SessionStateManager.has_network():
        return

    st.subheader("🎛️ Operation Mode")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "🤖 **Auto Mode**\n\nFully automated violation resolution",
            key="auto_mode",
            help="Automatically detect and resolve all violations without user intervention",
            use_container_width=True,
        ):
            SessionStateManager.set_mode("auto")
            st.rerun()

    with col2:
        if st.button(
            "💬 **Interactive Mode**\n\nStep-by-step with plan approval",
            key="interactive_mode",
            help="Review and approve each action plan before execution",
            use_container_width=True,
        ):
            SessionStateManager.set_mode("interactive")
            st.rerun()


def render_main_content():
    """Render the main content area based on current mode."""
    current_mode = SessionStateManager.get_current_mode()

    if current_mode == "auto":
        render_auto_mode()
    elif current_mode == "interactive":
        render_interactive_mode()
    # No else clause needed - network visualization is now at top level


def render_auto_mode():
    """Render automatic mode interface."""
    st.subheader("🤖 Automatic Mode")

    if st.button(
        "🚀 Start Automatic Resolution", type="primary", use_container_width=True
    ):
        with st.spinner("🔄 Running automated workflow..."):
            try:
                # Run automated workflow
                workflow = get_workflow()
                graph = workflow.compile()

                # Get current state
                state = SessionStateManager.get_workflow_state()

                # Execute workflow
                result = State(**graph.invoke(state))

                # Update session state
                SessionStateManager.update_workflow_result(result)

                st.success("✅ Automated workflow completed!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error in automated workflow: {e}")

    # Show progress if workflow is running
    if SessionStateManager.is_workflow_running():
        st.info("⏳ Workflow in progress...")
        # TODO: Add real-time progress tracking

    # Show workflow results if available
    if SessionStateManager.has_workflow_results():
        # Check if workflow just completed to auto-select results tab
        workflow_just_completed = st.session_state.get("workflow_just_completed", False)

        # Reset the flag after using it
        if workflow_just_completed:
            st.session_state.workflow_just_completed = False

        # Show workflow results
        render_workflow_results()
    elif not SessionStateManager.has_network():
        st.info("📁 Please select a network from the sidebar to begin analysis.")


def render_interactive_mode():
    """Render interactive mode interface."""
    st.subheader("💬 Interactive Mode")

    if not SessionStateManager.has_network():
        st.info("Please select a network to begin interactive analysis.")
        return

    # Initialize interactive state
    if "interactive_state" not in st.session_state:
        st.session_state.interactive_state = (
            "ready"  # ready, analyzing, planning, reviewing, executing
        )
        st.session_state.current_analysis = None
        st.session_state.proposed_plan = None
        st.session_state.execution_progress = None

    # Get current violations for analysis
    processor = NetworkDataProcessor()
    violations = processor.get_violations_summary()
    violations_data = processor.get_violations_data()

    # State machine for interactive workflow
    state = st.session_state.interactive_state

    if state == "ready":
        render_interactive_ready(violations, violations_data)
    elif state == "analyzing":
        render_interactive_analyzing()
    elif state == "planning":
        # render_interactive_planning()
        pass
    elif state == "reviewing":
        render_interactive_reviewing()
    elif state == "modifying":
        render_interactive_modifying()
    elif state == "executing":
        render_interactive_executing()

    # Network visualization is now shown at the top level


def render_interactive_ready(violations, violations_data):
    """Render the ready state of interactive mode."""
    total_violations = violations.get("total_violations", 0)

    if total_violations == 0:
        st.success("✅ Network has no violations - everything looks good!")
        return

    # Show violation summary
    st.info(f"🔍 Found {total_violations} violations that need attention")

    # Display violations in expandable sections
    col1, col2, col3 = st.columns(3)

    with col1:
        if violations.get("voltage_violations", 0) > 0:
            with st.expander(
                f"⚡ {violations['voltage_violations']} Voltage Violations",
                expanded=True,
            ):
                for v in violations_data.get("voltage_violations", []):
                    severity_color = {
                        "critical": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                    }.get(v["severity"], "⚪")
                    st.write(
                        f"{severity_color} Bus {v['bus_idx']}: {v['v_mag_pu']:.3f} p.u."
                    )

    with col2:
        if violations.get("thermal_violations", 0) > 0:
            with st.expander(
                f"🌡️ {violations['thermal_violations']} Thermal Violations",
                expanded=True,
            ):
                for t in violations_data.get("thermal_violations", []):
                    severity_color = {
                        "critical": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                    }.get(t["severity"], "⚪")
                    st.write(
                        f"{severity_color} {t['name']}: {t['loading_percent']:.1f}%"
                    )

    with col3:
        if violations.get("disconnected_buses", 0) > 0:
            with st.expander(
                f"🔌 {violations['disconnected_buses']} Disconnected Buses",
                expanded=True,
            ):
                for bus_id in violations_data.get("disconnected_buses", []):
                    st.write(f"🔴 Bus {bus_id}: No power flow")

    # Action buttons
    st.markdown("### 🎯 What would you like to do?")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔧 Generate Fix Plan", type="primary", use_container_width=True):
            st.session_state.interactive_state = "analyzing"
            st.rerun()

    with col2:
        if st.button("📊 Analyze in Detail", use_container_width=True):
            show_detailed_analysis(violations_data)

    with col3:
        if st.button("🔄 Refresh Analysis", use_container_width=True):
            st.rerun()


def render_interactive_analyzing():
    """Render the analyzing state."""
    st.info("🔄 Analyzing network and generating action plan...")

    with st.spinner("Running AI analysis..."):
        try:
            # Use the actual planner from the workflow
            from faraday.agents.workflow.nodes.planner import planner
            from faraday.agents.workflow.nodes.cache import cache_network

            # Get workflow state
            workflow_state = SessionStateManager.get_workflow_state()

            # Cache network first
            cached_state = cache_network(workflow_state)

            # Run planner to get actual action plan
            result_state = planner(cached_state)

            # Extract the proposed plan
            if result_state.iteration_results:
                latest_iteration = result_state.iteration_results[-1]
                proposed_plan = latest_iteration.executed_actions
                violations_before = latest_iteration.viola_before

                st.session_state.proposed_plan = proposed_plan
                st.session_state.violations_before = violations_before
                st.session_state.interactive_state = "reviewing"
                st.success("✅ Analysis complete! Generated action plan.")
                st.rerun()
            else:
                st.error("❌ Failed to generate action plan")
                st.session_state.interactive_state = "ready"

        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            st.session_state.interactive_state = "ready"


def render_interactive_reviewing():
    """Render the plan review state."""
    st.subheader("📋 Proposed Action Plan")

    plan = st.session_state.proposed_plan
    if not plan:
        st.error("No plan available")
        st.session_state.interactive_state = "ready"
        return

    st.info(
        f"💡 I've generated {len(plan)} actions to resolve the violations. Please review:"
    )

    # Display plan in a nice format
    for i, action in enumerate(plan, 1):
        action_name = action.get("name", "unknown")
        args = action.get("args", {})

        with st.expander(
            f"Action {i}: {action_name.replace('_', ' ').title()}", expanded=True
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                if action_name == "curtail_load":
                    st.write(f"📉 **Curtail Load**: {args.get('load_name', 'Unknown')}")
                    st.write(f"🎚️ **Reduction**: {args.get('curtail_percent', 0)}%")
                elif action_name == "update_switch_status":
                    status = "Close" if args.get("closed", False) else "Open"
                    st.write(
                        f"🔄 **Switch Operation**: {status} {args.get('switch_name', 'Unknown')}"
                    )
                elif action_name == "add_battery":
                    st.write(
                        f"🔋 **Add Battery**: Bus {args.get('bus_index', 'Unknown')}"
                    )
                    st.write(f"⚡ **Capacity**: {args.get('max_energy_kw', 1000)} kW")
                else:
                    st.write(f"🔧 **Action**: {action_name}")
                    st.write(f"📝 **Parameters**: {args}")

            with col2:
                # Action effectiveness (simplified)
                effectiveness = get_action_effectiveness(action)
                st.metric("Effectiveness", f"{effectiveness}%")

    # Action buttons for plan review
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Execute Plan", type="primary", use_container_width=True):
            st.session_state.interactive_state = "executing"
            st.rerun()

    with col2:
        if st.button("✏️ Modify Plan", use_container_width=True):
            st.session_state.interactive_state = "modifying"
            st.rerun()

    with col3:
        if st.button("❌ Generate New Plan", use_container_width=True):
            st.session_state.interactive_state = "analyzing"
            st.rerun()


def render_interactive_executing():
    """Render the execution state."""
    st.subheader("⚡ Executing Action Plan")

    plan = st.session_state.proposed_plan
    if not plan:
        st.error("No plan to execute")
        st.session_state.interactive_state = "ready"
        return

    # Initialize execution progress and save initial state
    if "execution_step" not in st.session_state:
        st.session_state.execution_step = 0
        st.session_state.execution_results = []

        # Save initial network state before any modifications
        import pandapower as pp
        import copy

        # Save initial network object (deep copy to preserve original state)
        if st.session_state.network is not None:
            st.session_state.initial_network_interactive = copy.deepcopy(
                st.session_state.network
            )

        # Save initial network file state
        workflow_state = SessionStateManager.get_workflow_state()
        if workflow_state and workflow_state.editing_network_file_path:
            initial_file_path = (
                workflow_state.work_dir + "/initial_network_interactive.json"
            )
            current_net = pp.from_json(workflow_state.editing_network_file_path)
            pp.to_json(current_net, initial_file_path)
            st.session_state.initial_network_file_path = initial_file_path

    current_step = st.session_state.execution_step
    total_steps = len(plan)

    # Show progress
    progress = current_step / total_steps if total_steps > 0 else 0
    st.progress(progress, text=f"Executing step {current_step + 1} of {total_steps}")

    if current_step < total_steps:
        # Execute current action
        current_action = plan[current_step]
        action_name = current_action.get("name", "unknown")

        st.info(f"🔄 Executing: {action_name.replace('_', ' ').title()}")

        # Execute the action using actual workflow tools
        with st.spinner(f"Applying {action_name}..."):
            try:
                # Import the actual tools
                from faraday.tools.pandapower import (
                    curtail_load,
                    update_switch_status,
                    add_battery,
                )

                # Get current network file path
                workflow_state = SessionStateManager.get_workflow_state()
                network_file_path = workflow_state.editing_network_file_path

                # Execute the actual action
                if action_name == "curtail_load":
                    curtail_load(
                        network_path=network_file_path,
                        load_name=current_action["args"]["load_name"],
                        curtail_percent=current_action["args"]["curtail_percent"],
                    )
                elif action_name == "update_switch_status":
                    update_switch_status(
                        network_path=network_file_path,
                        switch_name=current_action["args"]["switch_name"],
                        closed=current_action["args"]["closed"],
                    )
                elif action_name == "add_battery":
                    add_battery(
                        network_path=network_file_path,
                        bus_index=current_action["args"]["bus_index"],
                        max_energy_kw=current_action["args"]["max_energy_kw"],
                    )

                # Reload the updated network into session state
                import pandapower as pp

                updated_net = pp.from_json(network_file_path)
                st.session_state.network = updated_net

                # Mark as completed
                result = {
                    "action": current_action,
                    "status": "success",
                    "message": "Action completed successfully",
                }
                st.session_state.execution_results.append(result)
                st.session_state.execution_step += 1

                # Store executed actions for comparison view
                if "executed_actions_interactive" not in st.session_state:
                    st.session_state.executed_actions_interactive = []
                st.session_state.executed_actions_interactive.append(current_action)

                st.success(f"✅ Completed: {action_name}")
                st.rerun()

            except Exception as e:
                result = {
                    "action": current_action,
                    "status": "error",
                    "message": str(e),
                }
                st.session_state.execution_results.append(result)
                st.error(f"❌ Error executing {action_name}: {e}")

    else:
        # Execution complete
        st.success("🎉 All actions executed successfully!")

        # Show results summary
        success_count = len(
            [r for r in st.session_state.execution_results if r["status"] == "success"]
        )
        st.metric("Actions Completed", f"{success_count}/{total_steps}")

        # Check final violations
        processor = NetworkDataProcessor()
        final_violations = processor.get_violations_summary()
        final_total = final_violations.get("total_violations", 0)

        if final_total == 0:
            st.success("🎯 All violations resolved!")
        else:
            st.warning(
                f"⚠️ {final_total} violations remain - may need additional actions"
            )

        # Show comparison view with before/after results
        st.markdown("---")
        st.subheader("📊 Before vs After Comparison")

        # Create a mock workflow result to display the comparison
        # This simulates the workflow structure that ComparisonView expects
        if "violations_before" in st.session_state:
            try:
                from faraday.agents.workflow.state import (
                    IterationResult,
                    State,
                )
                from faraday.tools.pandapower import get_violations
                import pandapower as pp

                # Get current violations after execution
                workflow_state = SessionStateManager.get_workflow_state()
                current_net = pp.from_json(workflow_state.editing_network_file_path)
                current_violations = get_violations(current_net)

                # Get actual executed actions (only successful ones)
                successful_actions = [
                    result["action"]
                    for result in st.session_state.execution_results
                    if result["status"] == "success"
                ]

                # Create a mock workflow result for the comparison view
                mock_iteration = IterationResult(
                    iter=0,  # Required field
                    executed_actions=successful_actions,
                    viola_before=st.session_state.violations_before,
                    viola_after=current_violations,  # Use actual current violations
                )

                # Create a temporary workflow state for the comparison
                mock_workflow_result = State(
                    network_file_path=workflow_state.network_file_path,
                    org_network_copy_file_path=workflow_state.org_network_copy_file_path,
                    editing_network_file_path=workflow_state.editing_network_file_path,
                    work_dir=workflow_state.work_dir,
                    messages=workflow_state.messages,
                    iteration_results=[mock_iteration],
                    all_executed_actions=successful_actions,
                )

                # Temporarily store this for ComparisonView and update executed actions
                original_results = st.session_state.get("workflow_results")
                original_executed_actions = st.session_state.get("executed_actions")

                # Set temporary state for ComparisonView
                st.session_state.workflow_results = mock_workflow_result
                st.session_state.executed_actions = successful_actions

                # Render the comparison view
                comparison_view = ComparisonView()
                comparison_view.render()

                # Restore original results
                if original_results is not None:
                    st.session_state.workflow_results = original_results
                elif "workflow_results" in st.session_state:
                    del st.session_state.workflow_results

                if original_executed_actions is not None:
                    st.session_state.executed_actions = original_executed_actions
                elif "executed_actions" in st.session_state:
                    del st.session_state.executed_actions

            except Exception as e:
                st.warning(f"Could not display comparison view: {e}")
                st.info(
                    "💡 Comparison view is temporarily unavailable, but your actions were executed successfully."
                )

                # Debug information
                if st.checkbox("Show debug info", key="debug_comparison"):
                    st.write("**Debug Info:**")
                    st.write(
                        f"- Executed actions available: {'executed_actions_interactive' in st.session_state}"
                    )
                    st.write(
                        f"- Initial network saved: {'initial_network_interactive' in st.session_state}"
                    )
                    st.write(
                        f"- Violations before available: {'violations_before' in st.session_state}"
                    )
                    st.write(
                        f"- Current network available: {st.session_state.get('network') is not None}"
                    )
                    if "executed_actions_interactive" in st.session_state:
                        st.write(
                            f"- Number of executed actions: {len(st.session_state.executed_actions_interactive)}"
                        )
                    st.exception(e)

        # Reset for next iteration
        if st.button("🔄 Start New Analysis", type="primary", use_container_width=True):
            # Reset execution state
            keys_to_delete = [
                "execution_step",
                "execution_results",
                "executed_actions_interactive",
                "initial_network_interactive",
                "initial_network_file_path",
                "violations_before",
            ]
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.interactive_state = "ready"
            st.rerun()


def show_detailed_analysis(violations_data):
    """Show detailed violation analysis."""
    st.subheader("📊 Detailed Network Analysis")

    with st.expander("🔍 Violation Analysis", expanded=True):
        # Voltage violations details
        if violations_data.get("voltage_violations"):
            st.write("**Voltage Violations:**")
            df_voltage = pd.DataFrame(violations_data["voltage_violations"])
            st.dataframe(df_voltage, use_container_width=True)

        # Thermal violations details
        if violations_data.get("thermal_violations"):
            st.write("**Thermal Violations:**")
            df_thermal = pd.DataFrame(violations_data["thermal_violations"])
            st.dataframe(df_thermal, use_container_width=True)

        # Disconnected buses
        if violations_data.get("disconnected_buses"):
            st.write("**Disconnected Buses:**")
            st.write(violations_data["disconnected_buses"])


def render_interactive_modifying():
    """Render the plan modification state."""
    st.subheader("✏️ Modify Action Plan")

    plan = st.session_state.proposed_plan
    if not plan:
        st.error("No plan available to modify")
        st.session_state.interactive_state = "ready"
        return

    st.info(
        "💡 Edit, reorder, or remove actions from the plan below. You can also add new actions."
    )

    # Initialize modified plan if not exists
    if "modified_plan" not in st.session_state:
        st.session_state.modified_plan = plan.copy()

    modified_plan = st.session_state.modified_plan

    # Display current plan with edit options
    st.markdown("### 📋 Current Plan")

    if not modified_plan:
        st.warning("Plan is empty. Add actions below.")
    else:
        for i, action in enumerate(modified_plan):
            action_name = action.get("name", "unknown")
            args = action.get("args", {})

            with st.expander(
                f"Action {i + 1}: {action_name.replace('_', ' ').title()}",
                expanded=True,
            ):
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    # Display action details with edit capabilities
                    if action_name == "curtail_load":
                        st.write(
                            f"📉 **Curtail Load**: {args.get('load_name', 'Unknown')}"
                        )
                        new_percent = st.slider(
                            "Curtailment %",
                            min_value=1,
                            max_value=100,
                            value=int(args.get("curtail_percent", 10)),
                            key=f"curtail_{i}",
                        )
                        # Update the action with new value
                        if new_percent != args.get("curtail_percent", 10):
                            st.session_state.modified_plan[i]["args"][
                                "curtail_percent"
                            ] = new_percent

                    elif action_name == "update_switch_status":
                        switch_name = args.get("switch_name", "Unknown")
                        st.write(f"🔄 **Switch Operation**: {switch_name}")
                        new_status = st.selectbox(
                            "Switch Status",
                            ["Open", "Close"],
                            index=0 if not args.get("closed", False) else 1,
                            key=f"switch_{i}",
                        )
                        # Update the action with new value
                        new_closed = new_status == "Close"
                        if new_closed != args.get("closed", False):
                            st.session_state.modified_plan[i]["args"]["closed"] = (
                                new_closed
                            )

                    elif action_name == "add_battery":
                        st.write(
                            f"🔋 **Add Battery**: Bus {args.get('bus_index', 'Unknown')}"
                        )
                        new_capacity = st.slider(
                            "Battery Capacity (kW)",
                            min_value=100,
                            max_value=1000,
                            value=int(args.get("max_energy_kw", 1000)),
                            step=100,
                            key=f"battery_{i}",
                        )
                        # Update the action with new value
                        if new_capacity != args.get("max_energy_kw", 1000):
                            st.session_state.modified_plan[i]["args"][
                                "max_energy_kw"
                            ] = new_capacity

                    else:
                        st.write(f"🔧 **Action**: {action_name}")
                        st.write(f"📝 **Parameters**: {args}")

                with col2:
                    # Move action up/down
                    if st.button("⬆️", key=f"up_{i}", help="Move up", disabled=(i == 0)):
                        # Swap with previous action
                        (
                            st.session_state.modified_plan[i],
                            st.session_state.modified_plan[i - 1],
                        ) = (
                            st.session_state.modified_plan[i - 1],
                            st.session_state.modified_plan[i],
                        )
                        st.rerun()

                    if st.button(
                        "⬇️",
                        key=f"down_{i}",
                        help="Move down",
                        disabled=(i == len(modified_plan) - 1),
                    ):
                        # Swap with next action
                        (
                            st.session_state.modified_plan[i],
                            st.session_state.modified_plan[i + 1],
                        ) = (
                            st.session_state.modified_plan[i + 1],
                            st.session_state.modified_plan[i],
                        )
                        st.rerun()

                with col3:
                    # Remove action
                    if st.button("❌", key=f"remove_{i}", help="Remove action"):
                        st.session_state.modified_plan.pop(i)
                        st.rerun()

    # Add new action section
    st.markdown("### ➕ Add New Action")

    with st.expander("Add Action", expanded=False):
        action_type = st.selectbox(
            "Action Type",
            ["curtail_load", "update_switch_status", "add_battery"],
            key="new_action_type",
        )

        if action_type == "curtail_load":
            col1, col2 = st.columns(2)
            with col1:
                load_name = st.text_input("Load Name", key="new_load_name")
            with col2:
                curtail_percent = st.slider(
                    "Curtailment %", 1, 100, 10, key="new_curtail_percent"
                )

            if st.button("Add Load Curtailment"):
                new_action = {
                    "name": "curtail_load",
                    "args": {
                        "load_name": load_name,
                        "curtail_percent": curtail_percent,
                    },
                }
                st.session_state.modified_plan.append(new_action)
                st.success(f"Added load curtailment for {load_name}")
                st.rerun()

        elif action_type == "update_switch_status":
            col1, col2 = st.columns(2)
            with col1:
                switch_name = st.text_input("Switch Name", key="new_switch_name")
            with col2:
                switch_status = st.selectbox(
                    "Status", ["Open", "Close"], key="new_switch_status"
                )

            if st.button("Add Switch Operation"):
                new_action = {
                    "name": "update_switch_status",
                    "args": {
                        "switch_name": switch_name,
                        "closed": switch_status == "Close",
                    },
                }
                st.session_state.modified_plan.append(new_action)
                st.success(f"Added switch operation for {switch_name}")
                st.rerun()

        elif action_type == "add_battery":
            col1, col2 = st.columns(2)
            with col1:
                bus_index = st.number_input(
                    "Bus Index", min_value=0, key="new_bus_index"
                )
            with col2:
                battery_capacity = st.slider(
                    "Capacity (kW)", 100, 1000, 1000, 100, key="new_battery_capacity"
                )

            if st.button("Add Battery"):
                new_action = {
                    "name": "add_battery",
                    "args": {
                        "bus_index": int(bus_index),
                        "max_energy_kw": battery_capacity,
                    },
                }
                st.session_state.modified_plan.append(new_action)
                st.success(f"Added battery at Bus {bus_index}")
                st.rerun()

    # Action buttons for plan modification
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("✅ Save Changes", type="primary", use_container_width=True):
            # Update the main plan with modifications
            st.session_state.proposed_plan = st.session_state.modified_plan.copy()
            st.session_state.interactive_state = "reviewing"
            st.success("Plan updated successfully!")
            st.rerun()

    with col2:
        if st.button("🔄 Reset Plan", use_container_width=True):
            # Reset to original plan
            st.session_state.modified_plan = st.session_state.proposed_plan.copy()
            st.rerun()

    with col3:
        if st.button("❌ Cancel", use_container_width=True):
            # Cancel modifications and go back to reviewing
            if "modified_plan" in st.session_state:
                del st.session_state.modified_plan
            st.session_state.interactive_state = "reviewing"
            st.rerun()

    with col4:
        if st.button("🗑️ Clear All", use_container_width=True):
            # Clear all actions
            st.session_state.modified_plan = []
            st.rerun()


def get_action_effectiveness(action):
    """Calculate simplified action effectiveness score."""
    action_name = action.get("name", "")

    effectiveness_map = {
        "curtail_load": 75,
        "update_switch_status": 85,
        "add_battery": 80,
    }

    return effectiveness_map.get(action_name, 60)


# Removed unused function render_network_visualization() - functionality integrated into main content


def render_workflow_results():
    """Render workflow execution results."""
    st.subheader("📊 Workflow Results")

    # Comparison view
    comparison_view = ComparisonView()
    comparison_view.render()


def main():
    """Main dashboard application."""
    # Apply custom styling
    apply_custom_styles()

    # Initialize session state
    SessionStateManager.initialize()

    # Render header
    render_header()

    # Create layout
    _ = st.sidebar
    main_content = st.container()
    _ = st.sidebar

    # Render sidebar
    render_sidebar()

    # Main content area
    with main_content:
        # Network status overview
        render_network_status()

        # Network visualization (moved above mode selection)
        if SessionStateManager.has_network():
            st.subheader("🔍 Network Visualization")
            network_viz = NetworkVisualization()
            network_viz.render()

        # Mode selection
        render_mode_selection()

        # Main content based on mode
        render_main_content()


def run_dashboard():
    """Entry point for running the dashboard."""
    main()


if __name__ == "__main__":
    run_dashboard()
