"""
Status Panel Component

Real-time status monitoring and information display.
"""

import streamlit as st

from typing import Dict, Any

from faraday.dashboard.utils.session_state import SessionStateManager
from faraday.dashboard.utils.data_processing import NetworkDataProcessor
from faraday.tools.pandapower import get_voltage_thresholds


class StatusPanel:
    """Real-time status monitoring panel."""

    def __init__(self):
        self.processor = NetworkDataProcessor()

    def render(self):
        """Render the complete status panel."""
        # Network health status
        self._render_network_health()

        # Violation alerts
        self._render_violation_alerts()

        # Resource availability
        self._render_resource_status()

        # Activity log
        self._render_activity_log()

        # System status
        self._render_system_status()

    def _render_network_health(self):
        """Render network health overview."""
        st.subheader("🏥 Network Health")

        if not SessionStateManager.has_network():
            st.info("No network loaded")
            return

        try:
            metrics = self.processor.get_network_metrics()
            violations = self.processor.get_violations_summary()

            # Health score calculation
            total_elements = metrics.get("total_buses", 0) + metrics.get(
                "total_lines", 0
            )
            total_violations = violations.get("total_violations", 0)
            health_score = max(
                0, 100 - (total_violations / max(total_elements, 1)) * 100
            )

            # Health indicator
            if health_score >= 90:
                health_color = "🟢"
                health_status = "Excellent"
            elif health_score >= 75:
                health_color = "🟡"
                health_status = "Good"
            elif health_score >= 50:
                health_color = "🟠"
                health_status = "Warning"
            else:
                health_color = "🔴"
                health_status = "Critical"

            st.markdown(
                f"""
            <div class="status-card">
                <h4>{health_color} {health_status}</h4>
                <p>Health Score: <strong>{health_score:.1f}%</strong></p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Detailed metrics
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Buses", metrics.get("total_buses", 0))
                st.metric("Voltage Violations", violations.get("voltage_violations", 0))

            with col2:
                st.metric("Lines", metrics.get("total_lines", 0))
                st.metric("Thermal Violations", violations.get("thermal_violations", 0))

        except Exception as e:
            st.error(f"Error loading network health: {e}")

    def _render_violation_alerts(self):
        """Render violation alerts and warnings."""
        st.subheader("⚠️ Alerts")

        if not SessionStateManager.has_network():
            return

        try:
            violations_data = self.processor.get_violations_data()

            # Critical violations
            critical_voltage = [
                v
                for v in violations_data.get("voltage_violations", [])
                if v["v_mag_pu"] > 1.1 or v["v_mag_pu"] < 0.9
            ]
            critical_thermal = [
                t
                for t in violations_data.get("thermal_violations", [])
                if t["loading_percent"] > 120
            ]

            if critical_voltage or critical_thermal:
                st.error(
                    f"🚨 {len(critical_voltage + critical_thermal)} Critical Violations"
                )

                if critical_voltage:
                    st.write("**Critical Voltage Issues:**")
                    for v in critical_voltage[:3]:  # Show top 3
                        st.write(f"• Bus {v['bus_idx']}: {v['v_mag_pu']:.3f} p.u.")

                if critical_thermal:
                    st.write("**Critical Thermal Issues:**")
                    for t in critical_thermal[:3]:  # Show top 3
                        st.write(f"• {t['name']}: {t['loading_percent']:.1f}%")

            # Medium priority violations
            medium_violations = (
                len(violations_data.get("voltage_violations", []))
                + len(violations_data.get("thermal_violations", []))
                - len(critical_voltage)
                - len(critical_thermal)
            )

            if medium_violations > 0:
                st.warning(f"⚠️ {medium_violations} Medium Priority Violations")

            if not critical_voltage and not critical_thermal and medium_violations == 0:
                st.success("✅ No Active Violations")

        except Exception as e:
            st.error(f"Error loading violation alerts: {e}")

    def _render_resource_status(self):
        """Render controllable resource status."""
        st.subheader("🔧 Resources")

        if not SessionStateManager.has_network():
            return

        try:
            net = SessionStateManager.get_network()

            # Controllable resources summary
            curtailable_loads = len(
                [
                    load
                    for _, load in net.load.iterrows()
                    if load.get("curtailable", False)
                ]
            )
            controllable_switches = len(net.switch)
            available_battery_sites = len(
                net.bus
            )  # Simplified - any bus could have a battery

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Curtailable Loads",
                    curtailable_loads,
                    help="Loads that can be reduced to resolve violations",
                )

            with col2:
                st.metric(
                    "Switches",
                    controllable_switches,
                    help="Network switches available for topology control",
                )

            with col3:
                st.metric(
                    "Battery Sites",
                    available_battery_sites,
                    help="Potential locations for battery installation",
                )

            # Resource utilization
            current_plan = SessionStateManager.get_current_plan()
            if current_plan:
                plan_actions = [action["name"] for action in current_plan]
                curtail_actions = plan_actions.count("curtail_load")
                switch_actions = plan_actions.count("update_switch_status")
                battery_actions = plan_actions.count("add_battery")

                st.write("**Current Plan Usage:**")
                if curtail_actions > 0:
                    st.write(f"• {curtail_actions} load curtailment actions")
                if switch_actions > 0:
                    st.write(f"• {switch_actions} switch operations")
                if battery_actions > 0:
                    st.write(f"• {battery_actions} battery installations")

        except Exception as e:
            st.error(f"Error loading resource status: {e}")

    def _render_activity_log(self):
        """Render recent activity and execution log."""
        st.subheader("📝 Activity Log")

        # Get recent activities from session state
        activities = SessionStateManager.get_activity_log()

        if not activities:
            st.info("No recent activity")
            return

        # Display recent activities
        for activity in activities[-5:]:  # Show last 5 activities
            timestamp = activity.get("timestamp", "Unknown")
            action = activity.get("action", "Unknown")
            details = activity.get("details", "")
            status = activity.get("status", "info")

            if status == "success":
                st.success(f"✅ {timestamp}: {action} - {details}")
            elif status == "error":
                st.error(f"❌ {timestamp}: {action} - {details}")
            elif status == "warning":
                st.warning(f"⚠️ {timestamp}: {action} - {details}")
            else:
                st.info(f"ℹ️ {timestamp}: {action} - {details}")

        # Clear log button
        if st.button("🗑️ Clear Log", help="Clear activity log"):
            SessionStateManager.clear_activity_log()
            st.rerun()

    def _render_system_status(self):
        """Render system and workflow status."""
        st.subheader("⚙️ System Status")

        # Workflow status
        workflow_state = SessionStateManager.get_workflow_status()

        if workflow_state.get("is_running", False):
            st.info("🔄 Workflow Active")
            progress = workflow_state.get("progress", 0)
            st.progress(progress / 100)
            st.write(f"Current step: {workflow_state.get('current_step', 'Unknown')}")
        else:
            st.success("⏸️ Workflow Ready")

        # Session information
        session_info = SessionStateManager.get_session_info()

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Session Info:**")
            st.write(f"• Start time: {session_info.get('start_time', 'Unknown')}")
            st.write(f"• Actions executed: {session_info.get('actions_executed', 0)}")

        with col2:
            st.write("**Configuration:**")
            thresholds = get_voltage_thresholds()
            st.write(f"• V_max: {thresholds.v_max} p.u.")
            st.write(f"• V_min: {thresholds.v_min} p.u.")

        # Performance metrics
        if st.checkbox("Show Performance Metrics"):
            self._render_performance_metrics()

    def _render_performance_metrics(self):
        """Render system performance metrics."""
        st.write("**Performance Metrics:**")

        # Simulated performance data
        metrics = {
            "Average Response Time": "1.2s",
            "Success Rate": "94.5%",
            "Memory Usage": "156 MB",
            "Network Analysis Time": "0.8s",
            "Plan Generation Time": "0.4s",
        }

        for metric, value in metrics.items():
            st.write(f"• {metric}: {value}")

    def render_compact(self):
        """Render a compact version of the status panel."""
        if not SessionStateManager.has_network():
            st.info("📁 No network loaded")
            return

        try:
            violations = self.processor.get_violations_summary()
            total_violations = violations.get("total_violations", 0)

            if total_violations == 0:
                st.success("✅ No violations")
            else:
                st.error(f"🚨 {total_violations} violations")

            # Quick stats
            metrics = self.processor.get_network_metrics()
            st.write(
                f"📊 {metrics.get('total_buses', 0)} buses, {metrics.get('total_lines', 0)} lines"
            )

        except Exception as e:
            st.error(f"Status error: {e}")

    @staticmethod
    def render_progress_indicator(
        current_step: str, total_steps: int, current_step_num: int
    ):
        """Render a workflow progress indicator."""
        progress = (current_step_num / total_steps) * 100

        st.write(f"**Current Step:** {current_step}")
        st.progress(progress / 100)
        st.write(f"Step {current_step_num} of {total_steps}")

    @staticmethod
    def render_violation_summary_card(violations_data: Dict[str, Any]):
        """Render a compact violation summary card."""
        voltage_count = len(violations_data.get("voltage_violations", []))
        thermal_count = len(violations_data.get("thermal_violations", []))
        total_count = voltage_count + thermal_count

        if total_count == 0:
            st.markdown(
                """
            <div class="success-card">
                <h4>✅ Network Status: Healthy</h4>
                <p>No violations detected</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="violation-card">
                <h4>⚠️ Network Status: {total_count} Violations</h4>
                <p>Voltage: {voltage_count} | Thermal: {thermal_count}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
