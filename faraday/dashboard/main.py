"""
Faraday Dashboard V2 - Main Application

Advanced Streamlit dashboard for power grid violation analysis and resolution
with both automated and interactive modes.
"""

import json
import logging

from typing import Dict, Any

import streamlit as st
import pandas as pd

from faraday.dashboard.components.network_viz import NetworkVisualization
from faraday.dashboard.components.action_editor import ActionPlanEditor

from faraday.dashboard.components.comparison_view import ComparisonView
from faraday.dashboard.utils.session_state import SessionStateManager
from faraday.dashboard.utils.data_processing import NetworkDataProcessor
from faraday.agents.workflow.graph import get_workflow
from faraday.agents.workflow.state import State
from faraday.tools.pandapower import (
    get_voltage_thresholds,
    set_voltage_thresholds,
)

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
        st.subheader("📁 Network Management")
        uploaded_file = st.file_uploader(
            "Upload Network File",
            type=["json"],
            help="Upload a pandapower network JSON file",
        )

        if uploaded_file:
            try:
                network_data = json.load(uploaded_file)
                SessionStateManager.load_network(network_data, uploaded_file.name)
                st.success(f"✅ Loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")

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

        # Export options
        st.subheader("📤 Export")
        if st.button("Export Results"):
            # TODO: Implement export functionality
            st.info("Export functionality coming soon!")

        # Debug info (if in debug mode)
        if st.checkbox("🐛 Debug Mode"):
            st.subheader("Debug Information")
            st.json(st.session_state.to_dict())


def render_network_status():
    """Render network status overview."""
    if not SessionStateManager.has_network():
        st.info("📁 Please upload a network file to begin analysis.")
        return

    # Get network metrics
    processor = NetworkDataProcessor()
    metrics = processor.get_network_metrics()
    violations = processor.get_violations_summary()

    # Network health overview
    st.subheader("🏥 Network Health Overview")

    col1, col2, col3, col4 = st.columns(4)

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

    # Detailed violation information
    if (
        violations.get("voltage_violations", 0) > 0
        or violations.get("thermal_violations", 0) > 0
    ):
        st.markdown('<div class="violation-card">', unsafe_allow_html=True)
        st.warning(
            f"⚠️ Network has {violations.get('total_violations', 0)} violations requiring attention"
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

    col1, col2, col3 = st.columns(3)

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

    with col3:
        if st.button(
            "🔬 **Expert Mode**\n\nAdvanced manual controls",
            key="expert_mode",
            help="Full control over individual actions and parameters",
            use_container_width=True,
        ):
            SessionStateManager.set_mode("expert")
            st.rerun()


def render_main_content():
    """Render the main content area based on current mode."""
    current_mode = SessionStateManager.get_current_mode()

    if current_mode == "auto":
        render_auto_mode()
    elif current_mode == "interactive":
        render_interactive_mode()
    elif current_mode == "expert":
        render_expert_mode()
    else:
        render_network_visualization()


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

    # Always show network visualization if network is loaded
    if SessionStateManager.has_network():
        # Create tabs for better organization
        if SessionStateManager.has_workflow_results():
            # Check if workflow just completed to auto-select results tab
            workflow_just_completed = st.session_state.get(
                "workflow_just_completed", False
            )

            # Create tabs with conditional default selection
            if workflow_just_completed:
                # Auto-select Workflow Results tab after completion
                tab1, tab2 = st.tabs(
                    [
                        "📊 Workflow Results",
                        "🔍 Network Visualization",
                    ]
                )

                # Reset the flag after using it
                st.session_state.workflow_just_completed = False

                with tab1:
                    render_workflow_results()

                with tab2:
                    network_viz = NetworkVisualization()
                    network_viz.render()

            else:
                # Normal tab display
                tab1, tab2 = st.tabs(
                    ["🔍 Network Visualization", "📊 Workflow Results"]
                )

                with tab1:
                    network_viz = NetworkVisualization()
                    network_viz.render()

                with tab2:
                    render_workflow_results()
        else:
            # If no results yet, just show network visualization
            network_viz = NetworkVisualization()
            network_viz.render()
    elif SessionStateManager.has_workflow_results():
        # Show only results if no network is loaded
        render_workflow_results()


def render_interactive_mode():
    """Render interactive mode interface."""
    st.subheader("💬 Interactive Mode")

    # Chat interface
    chat_container = st.container()

    with chat_container:
        # Display chat messages
        messages = SessionStateManager.get_chat_messages()
        for message in messages:
            with st.chat_message(message["role"]):
                if message.get("plan_df") is not None:
                    st.dataframe(message["plan_df"], use_container_width=True)
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input(
        "What would you like me to do? (e.g., 'fix the violations')"
    ):
        SessionStateManager.add_chat_message("user", prompt)

        # Process user input
        response = process_chat_input(prompt)
        SessionStateManager.add_chat_message(
            "assistant", response["content"], response.get("plan_df")
        )

        st.rerun()


def render_expert_mode():
    """Render expert mode interface."""
    st.subheader("🔬 Expert Mode")

    # Action plan editor
    action_editor = ActionPlanEditor()
    action_editor.render()

    # Manual execution controls
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Analyze Network", use_container_width=True):
            # TODO: Implement detailed analysis
            st.info("Detailed analysis coming soon!")

    with col2:
        if st.button("⚡ Execute Plan", use_container_width=True):
            # TODO: Implement plan execution
            st.info("Plan execution coming soon!")


def render_network_visualization():
    """Render network visualization when no mode is selected."""
    if not SessionStateManager.has_network():
        return

    st.subheader("📊 Network Visualization")

    # Network visualization component
    network_viz = NetworkVisualization()
    network_viz.render()


def render_workflow_results():
    """Render workflow execution results."""
    st.subheader("📊 Workflow Results")

    # Comparison view
    comparison_view = ComparisonView()
    comparison_view.render()


def process_chat_input(user_input: str) -> Dict[str, Any]:
    """Process user input in interactive mode."""
    # Simplified chat processing - this would be expanded with actual LLM integration
    if "fix" in user_input.lower() and "violation" in user_input.lower():
        # Generate action plan
        # TODO: Integrate with actual planner
        plan = [
            {
                "name": "curtail_load",
                "args": {"load_name": "Load_1", "curtail_percent": 10},
            },
            {
                "name": "update_switch_status",
                "args": {"switch_name": "Switch_1", "closed": False},
            },
        ]

        plan_df = pd.DataFrame(
            [
                {
                    "Action": action["name"],
                    "Target": list(action["args"].values())[0],
                    "Parameters": str(action["args"]),
                }
                for action in plan
            ]
        )

        return {
            "content": "I've analyzed the network and generated an action plan. Please review and approve.",
            "plan_df": plan_df,
        }
    else:
        return {
            "content": "I can help you fix network violations. Try asking me to 'fix the violations'."
        }


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

        # Mode selection
        render_mode_selection()

        # Main content based on mode
        render_main_content()


def run_dashboard():
    """Entry point for running the dashboard."""
    main()


if __name__ == "__main__":
    run_dashboard()
