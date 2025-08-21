"""
Network Visualization Component

Interactive network plotting and analysis visualization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pandapower as pp
import pandapower.plotting.plotly as pplt
from typing import Dict
import numpy as np

from faraday.dashboard.utils.session_state import SessionStateManager
from faraday.dashboard.utils.data_processing import NetworkDataProcessor
from faraday.tools.pandapower import get_voltage_thresholds


class NetworkVisualization:
    """Interactive network visualization component."""

    def __init__(self):
        self.processor = NetworkDataProcessor()

    def render(self):
        """Render the complete network visualization interface."""
        if not SessionStateManager.has_network():
            st.info("📁 Please upload a network file to view visualization.")
            return

        # Visualization options
        self._render_visualization_controls()

        # Main network plot
        self._render_network_plot()

        # Additional analysis views
        self._render_analysis_tabs()

    def _render_visualization_controls(self):
        """Render visualization control options."""
        st.subheader("📊 Visualization Controls")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            plot_type = st.selectbox(
                "Plot Type",
                ["Network Topology", "Voltage Levels", "Power Flow", "Violations Only"],
                help="Select the type of network visualization",
            )

        with col2:
            color_scheme = st.selectbox(
                "Color Scheme",
                ["Default", "Voltage Based", "Loading Based", "Severity Based"],
                help="Choose coloring scheme for network elements",
            )

        with col3:
            show_labels = st.checkbox(
                "Show Labels",
                value=True,
                help="Display bus and line labels on the plot",
            )

        with col4:
            interactive_mode = st.checkbox(
                "Interactive Mode",
                value=True,
                help="Enable interactive pan, zoom, and hover",
            )

        # Store visualization settings
        SessionStateManager.set_viz_settings(
            {
                "plot_type": plot_type,
                "color_scheme": color_scheme,
                "show_labels": show_labels,
                "interactive_mode": interactive_mode,
            }
        )

    def _render_network_plot(self):
        """Render the main network plot."""
        try:
            net = SessionStateManager.get_network()
            if net is None:
                return

            settings = SessionStateManager.get_viz_settings()

            # Run power flow
            pp.runpp(net, algorithm="nr", calculate_voltage_angles=True)

            # Create the plot based on selected type
            if settings.get("plot_type") == "Violations Only":
                fig = self._create_violations_plot(net, settings)
            else:
                fig = self._create_standard_plot(net, settings)

            # Display the plot
            st.plotly_chart(fig, use_container_width=True, key="main_network_plot")

        except Exception as e:
            st.error(f"❌ Error creating network plot: {e}")
            if st.checkbox("Show debug details"):
                st.exception(e)

    def _create_standard_plot(self, net: pp.pandapowerNet, settings: Dict) -> go.Figure:
        """Create standard network topology plot using pandapower's pf_res_plotly directly."""
        try:
            # Use pandapower's pf_res_plotly directly
            fig = pplt.pf_res_plotly(net, auto_open=False)

            # Update layout with custom settings
            fig.update_layout(
                title=f"Network Topology - {settings.get('plot_type', 'Default')}",
                showlegend=True,
                height=600,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

            # Apply color customization if needed
            color_scheme = settings.get("color_scheme", "Default")
            if color_scheme != "Default":
                self._apply_color_scheme(fig, net, color_scheme)

            return fig

        except Exception as e:
            # Fallback to simple scatter plot
            return self._create_fallback_plot(net, str(e))

    def _create_violations_plot(
        self, net: pp.pandapowerNet, settings: Dict
    ) -> go.Figure:
        """Create a plot focused on violations only."""
        fig = go.Figure()

        # Get violations
        violations = self.processor.get_violations_data()
        _ = get_voltage_thresholds()

        # Plot buses with voltage violations
        if violations.get("voltage_violations"):
            voltage_viols = violations["voltage_violations"]
            bus_indices = [v["bus_idx"] for v in voltage_viols]
            voltages = [v["v_mag_pu"] for v in voltage_viols]

            # Get bus coordinates (simplified layout)
            x_coords = np.random.uniform(0, 10, len(bus_indices))
            y_coords = np.random.uniform(0, 10, len(bus_indices))

            # Color based on severity
            colors = ["red" if (v > 1.1 or v < 0.9) else "orange" for v in voltages]

            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers+text",
                    marker=dict(
                        size=15, color=colors, line=dict(width=2, color="black")
                    ),
                    text=[
                        f"Bus {idx}<br>V={v:.3f}"
                        for idx, v in zip(bus_indices, voltages)
                    ],
                    textposition="top center",
                    name="Voltage Violations",
                    hovertemplate="<b>Bus %{text}</b><br>Voltage: %{marker.color}<extra></extra>",
                )
            )

        # Plot thermal violations
        if violations.get("thermal_violations"):
            thermal_viols = violations["thermal_violations"]
            # Simplified representation of thermal violations
            for i, violation in enumerate(thermal_viols):
                fig.add_shape(
                    type="line",
                    x0=i * 2,
                    y0=0,
                    x1=i * 2 + 1,
                    y1=1,
                    line=dict(color="red", width=3),
                    name=f"Thermal: {violation['name']}",
                )

        fig.update_layout(
            title="Network Violations Overview",
            xaxis_title="Network Position",
            yaxis_title="Voltage Level (p.u.)",
            height=600,
            showlegend=True,
        )

        return fig

    def _create_fallback_plot(self, net: pp.pandapowerNet, error_msg: str) -> go.Figure:
        """Create a simple fallback plot when main plotting fails."""
        fig = go.Figure()

        # Simple bus representation
        n_buses = len(net.bus)
        x_coords = np.random.uniform(0, 10, n_buses)
        y_coords = np.random.uniform(0, 10, n_buses)

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker=dict(size=10, color="blue"),
                text=[f"Bus {i}" for i in range(n_buses)],
                name="Buses",
            )
        )

        fig.update_layout(
            title=f"Simplified Network View (Plot Error: {error_msg[:50]}...)",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            height=600,
        )

        return fig

    def _apply_color_scheme(
        self, fig: go.Figure, net: pp.pandapowerNet, color_scheme: str
    ) -> go.Figure:
        """Apply color scheme to the network plot."""
        try:
            # Note: Customizing pf_res_plotly output requires understanding its trace structure
            # For now, we keep the default pandapower coloring which already shows:
            # - Bus voltage levels
            # - Line loading levels
            # - Different colors for different element types

            # Future enhancement: modify specific traces based on color_scheme
            if color_scheme == "Voltage Based":
                # Could modify bus marker colors based on voltage levels
                pass
            elif color_scheme == "Loading Based":
                # Could modify line colors based on loading levels
                pass
            elif color_scheme == "Severity Based":
                # Could apply red/yellow/green based on violation severity
                pass

        except Exception as e:
            st.warning(f"Color scheme customization failed: {e}")

        return fig

    def _render_analysis_tabs(self):
        """Render additional analysis tabs."""
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Statistics", "⚡ Power Flow", "🔍 Violations", "📈 Trends"]
        )

        with tab1:
            self._render_statistics_tab()

        with tab2:
            self._render_power_flow_tab()

        with tab3:
            self._render_violations_tab()

        with tab4:
            self._render_trends_tab()

    def _render_statistics_tab(self):
        """Render network statistics."""
        st.subheader("Network Statistics")

        metrics = self.processor.get_network_metrics()
        violations = self.processor.get_violations_summary()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Buses", metrics.get("total_buses", 0))
            st.metric("Total Lines", metrics.get("total_lines", 0))
            st.metric("Total Transformers", metrics.get("total_transformers", 0))

        with col2:
            st.metric("Total Load (MW)", f"{metrics.get('total_load_mw', 0):.1f}")
            st.metric(
                "Total Generation (MW)", f"{metrics.get('total_generation_mw', 0):.1f}"
            )
            st.metric("Power Balance", f"{metrics.get('power_balance', 0):.1f}")

        with col3:
            st.metric("Voltage Violations", violations.get("voltage_violations", 0))
            st.metric("Thermal Violations", violations.get("thermal_violations", 0))
            st.metric("Health Score", f"{violations.get('health_score', 100):.1f}%")

    def _render_power_flow_tab(self):
        """Render power flow analysis."""
        st.subheader("Power Flow Analysis")

        try:
            net = SessionStateManager.get_network()
            if net is None:
                return

            # Bus results
            st.write("**Bus Voltages**")
            bus_results = pd.DataFrame(
                {
                    "Bus": net.bus.index,
                    "Name": net.bus.name,
                    "Voltage (p.u.)": net.res_bus.vm_pu.round(3),
                    "Angle (deg)": net.res_bus.va_degree.round(2),
                }
            )
            st.dataframe(bus_results, use_container_width=True)

            # Line results
            if len(net.line) > 0:
                st.write("**Line Loading**")
                line_results = pd.DataFrame(
                    {
                        "Line": net.line.index,
                        "Name": net.line.name,
                        "From Bus": net.line.from_bus,
                        "To Bus": net.line.to_bus,
                        "Loading (%)": net.res_line.loading_percent.round(1),
                        "P (MW)": net.res_line.p_from_mw.round(2),
                    }
                )
                st.dataframe(line_results, use_container_width=True)

        except Exception as e:
            st.error(f"Error displaying power flow results: {e}")

    def _render_violations_tab(self):
        """Render detailed violations analysis."""
        st.subheader("Violations Analysis")

        violations_data = self.processor.get_violations_data()

        # Voltage violations
        if violations_data.get("voltage_violations"):
            st.write("**🔋 Voltage Violations**")
            voltage_df = pd.DataFrame(violations_data["voltage_violations"])
            st.dataframe(voltage_df, use_container_width=True)

            # Voltage distribution chart
            if len(voltage_df) > 0:
                fig_voltage = px.histogram(
                    voltage_df, x="v_mag_pu", title="Voltage Distribution", nbins=20
                )
                thresholds = get_voltage_thresholds()
                fig_voltage.add_vline(
                    x=thresholds.v_min,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Min Limit",
                )
                fig_voltage.add_vline(
                    x=thresholds.v_max,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Max Limit",
                )
                st.plotly_chart(fig_voltage, use_container_width=True)

        # Thermal violations
        if violations_data.get("thermal_violations"):
            st.write("**🌡️ Thermal Violations**")
            thermal_df = pd.DataFrame(violations_data["thermal_violations"])
            st.dataframe(thermal_df, use_container_width=True)

            # Loading distribution chart
            if len(thermal_df) > 0:
                fig_thermal = px.bar(
                    thermal_df,
                    x="name",
                    y="loading_percent",
                    title="Line Loading Violations",
                    color="loading_percent",
                    color_continuous_scale="Reds",
                )
                fig_thermal.add_hline(
                    y=100,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="100% Limit",
                )
                st.plotly_chart(fig_thermal, use_container_width=True)

    def _render_trends_tab(self):
        """Render trends and historical data."""
        st.subheader("Trends Analysis")

        # For now, show placeholder content
        st.info(
            "📈 Historical trend analysis will be available once multiple iterations are completed."
        )

        # Placeholder charts
        if st.checkbox("Show sample trend data"):
            # Generate sample data
            dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
            voltage_violations = np.random.poisson(2, 30)
            thermal_violations = np.random.poisson(1, 30)

            trend_df = pd.DataFrame(
                {
                    "Date": dates,
                    "Voltage Violations": voltage_violations,
                    "Thermal Violations": thermal_violations,
                }
            )

            fig_trends = px.line(
                trend_df,
                x="Date",
                y=["Voltage Violations", "Thermal Violations"],
                title="Sample: Violation Trends Over Time",
            )
            st.plotly_chart(fig_trends, use_container_width=True)
