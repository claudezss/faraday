import json
from enum import Enum

import pandas as pd
import pandapower as pp
import streamlit as st
from dataclasses import dataclass, field

from energiq_agent import ROOT_DIR
from energiq_agent.agents.graph import get_workflow, State

# --- Initialization ---
wf = get_workflow()
graph = wf.compile()

WORKSPACE = ROOT_DIR.parent / "workspace"
WORKSPACE_NETWORKS = WORKSPACE / "networks"
WORKSPACE_NETWORKS.mkdir(parents=True, exist_ok=True)


class Step(str, Enum):
    VIEW = "view"
    PLAN = "plan"
    EXECUTE = "execute"
    DONE = "done"


@dataclass
class AppSessionState:
    step: Step = Step.VIEW
    action_plan: list = field(default_factory=list)
    summary: str = ""
    line_violations: pd.DataFrame = None
    voltage_violations: pd.DataFrame = None
    state: State = None
    net: pp.pandapowerNet = None
    iter: int = 0

    # UI control
    show_plan_section: bool = False
    show_execute_section: bool = False


def initialize_session_state():
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppSessionState()


initialize_session_state()
app_state = st.session_state.app_state

# --- UI Components ---
st.set_page_config(layout="wide")
st.title("⚡ Interactive Power Grid Agent")

# File uploader
uploaded_file = st.file_uploader("Upload pandapower JSON network", type=["json"])

if uploaded_file and app_state.net is None:
    network_data = json.load(uploaded_file)
    net = pp.from_json_string(json.dumps(network_data))
    network_file_path = WORKSPACE_NETWORKS / "network.json"
    pp.to_json(net, str(network_file_path))
    app_state.net = net
    app_state.state = State(
        network_file_path=str(network_file_path),
        editing_network_file_path=str(network_file_path),
        work_dir=str(WORKSPACE_NETWORKS.absolute()),
        messages=[],
        network=net,
        iter=0,
    )

if app_state.net is not None:
    st.subheader("📊 Network Overview")
    net = app_state.net
    st.write(
        f"Buses: {len(net.bus)}, Lines: {len(net.line)}, Switches: {len(net.switch)}"
    )

    with st.spinner("Checking for violations..."):
        pp.runpp(net)
        app_state.line_violations = net.res_line[net.res_line.loading_percent > 100]
        app_state.voltage_violations = net.res_bus[
            (net.res_bus.vm_pu > 1.05) | (net.res_bus.vm_pu < 0.95)
        ]

    with st.expander("Initial Network Plot"):
        fig = pp.plotting.plotly.pf_res_plotly(net, auto_open=False)
        st.plotly_chart(fig, use_container_width=True)

    st.write("Initial Thermal Violations:")
    st.dataframe(app_state.line_violations)
    st.write("Initial Voltage Violations:")
    st.dataframe(app_state.voltage_violations)

    st.subheader(f"🧠 Planner (Iteration {app_state.iter + 1})")

    if st.button("Generate and Execute Plan"):
        app_state.step = Step.PLAN
        app_state.iter += 1

        with st.spinner(f"Running Planner and Executor (Attempt {app_state.iter})..."):
            final_state = graph.invoke(app_state.state)

            app_state.state.update(final_state)
            app_state.action_plan = final_state.get("action_plan", [])
            app_state.summary = final_state.get("summary", "")
            app_state.show_execute_section = True

    if app_state.show_execute_section:
        st.subheader("📝 Executed Action Plan")
        st.json(app_state.action_plan)

        st.subheader("⚙️ Network Status After Execution")

        final_net = pp.from_json(app_state.state["editing_network_file_path"])
        pp.runpp(final_net)

        line_violations_after = final_net.res_line[
            final_net.res_line.loading_percent > 100
        ]
        voltage_violations_after = final_net.res_bus[
            (final_net.res_bus.vm_pu > 1.05) | (final_net.res_bus.vm_pu < 0.95)
        ]

        st.write("Thermal Violations After Execution:")
        st.dataframe(line_violations_after)
        st.write("Voltage Violations After Execution:")
        st.dataframe(voltage_violations_after)

        with st.expander("Final Network Plot"):
            fig = pp.plotting.plotly.pf_res_plotly(final_net, auto_open=False)
            st.plotly_chart(fig, use_container_width=True)

        if app_state.summary:
            st.subheader("✅ Summary of Actions")
            st.markdown(app_state.summary)
            app_state.step = Step.DONE

        elif line_violations_after.empty and voltage_violations_after.empty:
            st.success("All violations resolved successfully!")
            app_state.step = Step.DONE

        elif app_state.state["iter"] >= 3:
            st.warning("Max iterations reached. Could not resolve all violations.")
            app_state.step = Step.DONE
        else:
            st.info(
                "Violations remain. Click 'Generate and Execute Plan' to try again."
            )
