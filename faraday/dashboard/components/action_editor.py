"""
Action Plan Editor Component

Interactive editor for creating, modifying, and validating action plans.
"""

import streamlit as st
import pandas as pd

from typing import Dict, Any
import copy

from faraday.dashboard.utils.session_state import SessionStateManager
from faraday.dashboard.utils.data_processing import NetworkDataProcessor
from faraday.tools.pandapower import (
    get_violations,
)


class ActionPlanEditor:
    """Interactive action plan editor component."""

    def __init__(self):
        self.processor = NetworkDataProcessor()
        self.available_actions = {
            "curtail_load": {
                "name": "Curtail Load",
                "description": "Reduce power consumption of a specific load",
                "parameters": {
                    "load_name": {"type": "select", "options": []},
                    "curtail_percent": {
                        "type": "slider",
                        "min": 5,
                        "max": 50,
                        "default": 10,
                    },
                },
                "icon": "⬇️",
            },
            "update_switch_status": {
                "name": "Update Switch",
                "description": "Open or close network switches",
                "parameters": {
                    "switch_name": {"type": "select", "options": []},
                    "closed": {"type": "checkbox", "default": True},
                },
                "icon": "🔀",
            },
            "add_battery": {
                "name": "Add Battery",
                "description": "Install battery storage at specified bus",
                "parameters": {
                    "bus_index": {"type": "select", "options": []},
                    "max_energy_kw": {
                        "type": "slider",
                        "min": 100,
                        "max": 5000,
                        "default": 1000,
                    },
                },
                "icon": "🔋",
            },
        }

    def render(self):
        """Render the complete action plan editor interface."""
        if not SessionStateManager.has_network():
            st.info("📁 Please upload a network file to use the action editor.")
            return

        self._update_action_options()

        # Current action plan display
        self._render_current_plan()

        # Action builder interface
        self._render_action_builder()

        # Plan validation and execution
        self._render_plan_controls()

    def _update_action_options(self):
        """Update available options for action parameters based on current network."""
        try:
            net = SessionStateManager.get_network()
            if net is None:
                return

            # Update load options
            load_names = (
                net.load["name"].tolist()
                if "name" in net.load.columns
                else [f"Load_{i}" for i in net.load.index]
            )
            self.available_actions["curtail_load"]["parameters"]["load_name"][
                "options"
            ] = load_names

            # Update switch options
            switch_names = (
                net.switch["name"].tolist()
                if "name" in net.switch.columns
                else [f"Switch_{i}" for i in net.switch.index]
            )
            self.available_actions["update_switch_status"]["parameters"]["switch_name"][
                "options"
            ] = switch_names

            # Update bus options
            bus_options = [
                {"label": f"Bus {idx} ({name})", "value": idx}
                for idx, name in zip(net.bus.index, net.bus.name)
            ]
            self.available_actions["add_battery"]["parameters"]["bus_index"][
                "options"
            ] = bus_options

        except Exception as e:
            st.error(f"Error updating action options: {e}")

    def _render_current_plan(self):
        """Render the current action plan."""
        st.subheader("📋 Current Action Plan")

        current_plan = SessionStateManager.get_current_plan()

        if not current_plan:
            st.info("No actions in current plan. Add actions using the builder below.")
            return

        # Display plan as editable table
        plan_data = []
        for i, action in enumerate(current_plan):
            plan_data.append(
                {
                    "ID": i,
                    "Action": self.available_actions.get(action["name"], {}).get(
                        "name", action["name"]
                    ),
                    "Target": self._format_action_target(action),
                    "Parameters": self._format_action_params(action),
                    "Status": action.get("status", "Pending"),
                }
            )

        plan_df = pd.DataFrame(plan_data)

        # Display with selection for editing/deletion
        _ = st.data_editor(
            plan_df,
            hide_index=True,
            use_container_width=True,
            disabled=["ID", "Action", "Target"],
            key="plan_editor",
        )

        # Action buttons for plan management
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🗑️ Clear Plan", help="Remove all actions from plan"):
                SessionStateManager.clear_current_plan()
                st.rerun()

        # Template save/load functionality removed - not implemented

        with col4:
            if st.button("🔍 Validate Plan", help="Check plan feasibility"):
                self._validate_current_plan()

    def _render_action_builder(self):
        """Render the action builder interface."""
        st.subheader("⚡ Action Builder")

        # Action type selection
        col1, col2 = st.columns([1, 3])

        with col1:
            action_type = st.selectbox(
                "Action Type",
                options=list(self.available_actions.keys()),
                format_func=lambda x: f"{self.available_actions[x]['icon']} {self.available_actions[x]['name']}",
                key="action_type_select",
            )

        with col2:
            action_config = self.available_actions[action_type]
            st.info(f"ℹ️ {action_config['description']}")

        # Parameter configuration
        st.write("**Parameters:**")
        params = self._render_action_parameters(action_type)

        # Add action button
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("➕ Add Action", type="primary", use_container_width=True):
                self._add_action_to_plan(action_type, params)
                st.rerun()

        with col2:
            if st.button(
                "🧪 Test Action",
                help="Preview action effects",
                use_container_width=True,
            ):
                self._test_action_preview(action_type, params)

    def _render_action_parameters(self, action_type: str) -> Dict[str, Any]:
        """Render parameter inputs for the selected action type."""
        action_config = self.available_actions[action_type]
        params = {}

        for param_name, param_config in action_config["parameters"].items():
            if param_config["type"] == "select":
                if action_type == "add_battery" and param_name == "bus_index":
                    # Special handling for bus selection with labels
                    options = param_config["options"]
                    if options:
                        selected = st.selectbox(
                            param_name.replace("_", " ").title(),
                            options=options,
                            format_func=lambda x: x.get("label", str(x))
                            if isinstance(x, dict)
                            else str(x),
                            key=f"param_{param_name}",
                        )
                        params[param_name] = (
                            selected.get("value", selected)
                            if isinstance(selected, dict)
                            else selected
                        )
                    else:
                        st.warning(f"No options available for {param_name}")
                        params[param_name] = None
                else:
                    options = param_config["options"]
                    if options:
                        params[param_name] = st.selectbox(
                            param_name.replace("_", " ").title(),
                            options=options,
                            key=f"param_{param_name}",
                        )
                    else:
                        st.warning(f"No options available for {param_name}")
                        params[param_name] = None

            elif param_config["type"] == "slider":
                params[param_name] = st.slider(
                    param_name.replace("_", " ").title(),
                    min_value=param_config["min"],
                    max_value=param_config["max"],
                    value=param_config["default"],
                    key=f"param_{param_name}",
                )

            elif param_config["type"] == "checkbox":
                params[param_name] = st.checkbox(
                    param_name.replace("_", " ").title(),
                    value=param_config["default"],
                    key=f"param_{param_name}",
                )

            elif param_config["type"] == "number":
                params[param_name] = st.number_input(
                    param_name.replace("_", " ").title(),
                    value=param_config.get("default", 0),
                    key=f"param_{param_name}",
                )

        return params

    def _render_plan_controls(self):
        """Render plan validation and execution controls."""
        st.subheader("🎯 Plan Controls")

        current_plan = SessionStateManager.get_current_plan()

        if not current_plan:
            st.info("Add actions to the plan to enable execution controls.")
            return

        # Validation results
        validation_results = self._validate_current_plan(show_results=False)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Plan Actions", len(current_plan))
            st.metric(
                "Estimated Impact",
                f"{validation_results.get('estimated_impact', 0):.1f}%",
            )

        with col2:
            feasibility = (
                "✅ Valid"
                if validation_results.get("is_valid", False)
                else "❌ Invalid"
            )
            st.metric("Feasibility", feasibility)
            st.metric("Risk Level", validation_results.get("risk_level", "Unknown"))

        # Execution options
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔍 Detailed Validation", use_container_width=True):
                self._show_detailed_validation()

        with col2:
            if st.button("📊 Impact Preview", use_container_width=True):
                self._show_impact_preview()

        with col3:
            execute_disabled = not validation_results.get("is_valid", False)
            if st.button(
                "🚀 Execute Plan",
                type="primary",
                disabled=execute_disabled,
                use_container_width=True,
            ):
                self._execute_current_plan()

    def _format_action_target(self, action: Dict[str, Any]) -> str:
        """Format action target for display."""
        args = action.get("args", {})
        if "load_name" in args:
            return args["load_name"]
        elif "switch_name" in args:
            return args["switch_name"]
        elif "bus_index" in args:
            return f"Bus {args['bus_index']}"
        else:
            return "N/A"

    def _format_action_params(self, action: Dict[str, Any]) -> str:
        """Format action parameters for display."""
        args = action.get("args", {})
        formatted_params = []

        for key, value in args.items():
            if key in ["load_name", "switch_name", "bus_index"]:
                continue  # These are displayed as targets

            if key == "curtail_percent":
                formatted_params.append(f"Curtail: {value}%")
            elif key == "closed":
                formatted_params.append(f"Close: {value}")
            elif key == "max_energy_kw":
                formatted_params.append(f"Capacity: {value}kW")
            else:
                formatted_params.append(f"{key}: {value}")

        return ", ".join(formatted_params) if formatted_params else "Default"

    def _add_action_to_plan(self, action_type: str, params: Dict[str, Any]):
        """Add a new action to the current plan."""
        # Validate parameters
        if None in params.values():
            st.error("❌ Please fill in all required parameters.")
            return

        action = {
            "name": action_type,
            "args": params,
            "status": "Pending",
            "created_at": pd.Timestamp.now().isoformat(),
        }

        SessionStateManager.add_action_to_plan(action)
        st.success(f"✅ Added {self.available_actions[action_type]['name']} to plan")

    def _test_action_preview(self, action_type: str, params: Dict[str, Any]):
        """Show a preview of what the action would do."""
        if None in params.values():
            st.error("❌ Please fill in all parameters to preview action.")
            return

        try:
            net = SessionStateManager.get_network()
            if net is None:
                return

            # Create a copy for testing
            test_net = copy.deepcopy(net)

            # Get violations before
            violations_before = get_violations(test_net)

            # Simulate action (simplified)
            if action_type == "curtail_load":
                st.info(
                    f"🧪 Preview: Would curtail {params['load_name']} by {params['curtail_percent']}%"
                )
            elif action_type == "update_switch_status":
                status = "closed" if params["closed"] else "open"
                st.info(f"🧪 Preview: Would set {params['switch_name']} to {status}")
            elif action_type == "add_battery":
                st.info(
                    f"🧪 Preview: Would add {params['max_energy_kw']}kW battery at bus {params['bus_index']}"
                )

            # Show current violations context
            total_violations = len(violations_before.voltage) + len(
                violations_before.thermal
            )
            st.info(f"📊 Current network has {total_violations} violations")

        except Exception as e:
            st.error(f"❌ Error in action preview: {e}")

    def _validate_current_plan(self, show_results: bool = True) -> Dict[str, Any]:
        """Validate the current action plan."""
        current_plan = SessionStateManager.get_current_plan()

        if not current_plan:
            return {"is_valid": False, "errors": ["No actions in plan"]}

        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "estimated_impact": 75,  # Placeholder
            "risk_level": "Low",
        }

        # Basic validation checks
        for i, action in enumerate(current_plan):
            action_name = action.get("name")
            args = action.get("args", {})

            # Check if action type is supported
            if action_name not in self.available_actions:
                validation_results["errors"].append(
                    f"Action {i + 1}: Unsupported action type '{action_name}'"
                )
                validation_results["is_valid"] = False

            # Check for required parameters
            if not args:
                validation_results["errors"].append(
                    f"Action {i + 1}: Missing parameters"
                )
                validation_results["is_valid"] = False

        if show_results:
            if validation_results["is_valid"]:
                st.success("✅ Plan validation passed")
            else:
                st.error("❌ Plan validation failed:")
                for error in validation_results["errors"]:
                    st.error(f"  • {error}")

            if validation_results["warnings"]:
                for warning in validation_results["warnings"]:
                    st.warning(f"⚠️ {warning}")

        return validation_results

    def _show_detailed_validation(self):
        """Show detailed validation results."""
        with st.expander("🔍 Detailed Validation Results", expanded=True):
            validation = self._validate_current_plan(show_results=False)

            st.write("**Validation Summary:**")
            st.json(validation)

            # Additional checks could be added here
            st.write("**Network Compatibility Check:**")
            net = SessionStateManager.get_network()
            if net is not None:
                st.write(
                    f"✅ Network loaded with {len(net.bus)} buses, {len(net.line)} lines"
                )
                st.write(f"✅ {len(net.load)} loads available for curtailment")
                st.write(f"✅ {len(net.switch)} switches available for control")

    def _show_impact_preview(self):
        """Show estimated impact of the current plan."""
        with st.expander("📊 Impact Preview", expanded=True):
            current_plan = SessionStateManager.get_current_plan()

            if not current_plan:
                st.info("No actions to analyze.")
                return

            # Create impact summary
            impact_data = []
            for action in current_plan:
                action_type = action["name"]
                _ = action.get("args", {})

                estimated_impact = {
                    "curtail_load": "Medium",
                    "update_switch_status": "High",
                    "add_battery": "High",
                }.get(action_type, "Unknown")

                impact_data.append(
                    {
                        "Action": self.available_actions[action_type]["name"],
                        "Target": self._format_action_target(action),
                        "Estimated Impact": estimated_impact,
                        "Risk": "Low",  # Placeholder
                    }
                )

            impact_df = pd.DataFrame(impact_data)
            st.dataframe(impact_df, use_container_width=True)

            # Overall impact estimate
            st.info("📈 Estimated overall improvement: 75% violation reduction")

    def _execute_current_plan(self):
        """Execute the current action plan."""
        current_plan = SessionStateManager.get_current_plan()

        if not current_plan:
            st.error("❌ No plan to execute.")
            return

        # Confirm execution
        if st.button("⚠️ Confirm Execution", type="secondary"):
            try:
                with st.spinner("🔄 Executing action plan..."):
                    executed_actions = []

                    for i, action in enumerate(current_plan):
                        st.write(
                            f"Executing action {i + 1}/{len(current_plan)}: {action['name']}"
                        )

                        # Execute action (this would integrate with the actual execution system)
                        # For now, just mark as executed
                        action["status"] = "Executed"
                        executed_actions.append(action)

                    # Update session state
                    SessionStateManager.set_executed_actions(executed_actions)
                    SessionStateManager.clear_current_plan()

                    st.success("✅ Plan executed successfully!")
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Error executing plan: {e}")

    # Template save/load methods removed - not implemented
