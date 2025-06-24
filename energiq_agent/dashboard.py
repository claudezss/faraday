from enum import Enum

import streamlit as st
import json
import pandas as pd
import pandapower as pp
from langgraph.types import Command
from dataclasses import dataclass, field

from energiq_agent.agents.graph import get_workflow, State
from energiq_agent import ROOT_DIR

workflow = get_workflow()
graph = workflow.compile()

WORKSPACE = ROOT_DIR.parent / "workspace"

WORKSPACE_NETWORKS = WORKSPACE / "networks"

WORKSPACE_NETWORKS.mkdir(parents=True, exist_ok=True)


class Step(str, Enum):
    view = "view"
    plan = "plan"
    execute = "execute"
    critic = "critic"
    done = "done"


@dataclass
class SessionState:
    step: Step = Step.view
    action_plan: str = ""
    action_log: list = field(default_factory=list)
    line_violations: pd.DataFrame = None
    voltage_violations: pd.DataFrame = None
    line_violations_after_action: pd.DataFrame = None
    voltage_violations_after_action: pd.DataFrame = None
    state: State = None
    net: pp.pandapowerNet = None
    plan_generate_button_text: str = "Generate"
    feedback: str = ""
    iter: int = 0
    org_line_violation_count = 0
    org_voltage_violation_count = 0
    line_violation_count = 0
    voltage_violation_count = 0

    # block status
    show_plan_section: bool = False
    show_execute_section: bool = False
    show_critic_section: bool = False


def initialize(session_state):
    if not session_state:
        return SessionState()
    else:
        return session_state


st.session_state = initialize(st.session_state)


@st.dialog("Action Log")
def action_log_dialog():
    st.write(st.session_state.action_log)


st.sidebar.write("## Violations")
col1, col2 = st.sidebar.columns(2)
col1.metric(
    label="Thermal",
    value=st.session_state.line_violation_count,
    delta=st.session_state.line_violation_count
    - st.session_state.org_line_violation_count,
    delta_color="inverse",
)
st.session_state.org_line_violation_count = st.session_state.line_violation_count
col2.metric(
    label="Voltage",
    value=st.session_state.voltage_violation_count,
    delta=st.session_state.voltage_violation_count
    - st.session_state.org_voltage_violation_count,
    delta_color="inverse",
)

st.sidebar.write("## Action Logs")


@st.dialog("Action Logs")
def show_action_logs():
    try:
        with open(WORKSPACE_NETWORKS / "action_log.json", "r") as f:
            logs = json.load(f)
        for item in logs:
            st.write(item)
    except FileNotFoundError:
        st.write("No logs found")


act_logs = st.sidebar.button("Action Logs")

if act_logs:
    show_action_logs()

st.set_page_config(layout="wide")
st.title("⚡ Interactive Power Grid Agent")

# Upload grid JSON
uploaded_file = st.file_uploader("Upload pandapower JSON network", type=["json"])

if uploaded_file and st.session_state.net is None:
    network_data = json.load(uploaded_file)
    net = pp.from_json_string(json.dumps(network_data))
    pp.to_json(net, WORKSPACE_NETWORKS / "network.json")
    st.session_state.net = net

# State: Network Loaded
if st.session_state.net is not None:
    st.subheader("📊 Network Overview")
    net = st.session_state.net
    st.write(
        f"🧵 Buses: {len(net.bus)}, 🔌 Lines: {len(net.line)}, 🔁 Switches: {len(net.switch)}"
    )

    with st.spinner("Checking violations. Please wait...", show_time=True):
        pp.runpp(net)
        st.session_state.line_violations = net.res_line[
            net.res_line.loading_percent > 100
        ]
        st.session_state.voltage_violations = net.res_bus[
            (net.res_bus.vm_pu > 1.05) | (net.res_bus.vm_pu < 0.95)
        ]
        st.session_state.org_line_violation_count = len(
            st.session_state.line_violations
        )
        st.session_state.org_voltage_violation_count = len(
            st.session_state.voltage_violations
        )
        st.session_state.line_violation_count = len(st.session_state.line_violations)
        st.session_state.voltage_violation_count = len(
            st.session_state.voltage_violations
        )

    with st.spinner("Plotting Network. Please wait...", show_time=True):
        with st.expander("Network Plot (click to expand)"):
            fig = pp.plotting.plotly.pf_res_plotly(net, auto_open=False)
            st.plotly_chart(fig, use_container_width=False)

    if st.session_state.line_violations is not None:
        st.write("Thermal Violations (if any):")
        st.dataframe(st.session_state.line_violations)

    if st.session_state.voltage_violations is not None:
        st.write("Voltage Violations (if any):")
        st.dataframe(st.session_state.voltage_violations)

    # Planner Step
    st.subheader(f"🧠 Planner Suggestions (iter {st.session_state.iter})")

    if st.session_state.step == "view":
        if st.button("Start Planning"):
            st.session_state.step = Step.plan
            st.session_state.show_plan_section = True
            st.rerun()

    if st.session_state.show_plan_section:
        st.session_state.state = State(
            network_file_path="network.json",
            human_guidance="",
            editing_network_file_path=str(WORKSPACE_NETWORKS / "network.json"),
            messages=[],
            network=pp.from_json(WORKSPACE_NETWORKS / "network.json"),
            work_dir=str(WORKSPACE_NETWORKS.absolute()),
            action_plan="",
            violation_before_action=None,
            violation_after_action=None,
            feedback=st.session_state.feedback,
            log=st.session_state.action_log,
            iter=st.session_state.iter,
        )

        with st.form("planner_form"):
            st.write("Inputs")
            helper = st.text_area(
                "Provide a helper text to guide the planner (Optional)"
            )
            if st.session_state.feedback:
                critic_feedback = st.text_area(
                    "Critic Feedback", value=st.session_state.feedback, height=400
                )
            else:
                critic_feedback = ""

            submit = st.form_submit_button(
                st.session_state.plan_generate_button_text,
            )

            if submit:
                st.session_state.state["human_guidance"] = helper
                st.session_state.state["feedback"] = critic_feedback

                with st.spinner("Generating action plan", show_time=True):
                    state_update: Command = graph.nodes["planner"].invoke(
                        st.session_state.state
                    )
                st.session_state.action_plan = state_update.update["action_plan"]
                st.session_state.state.update(state_update.update)
                st.session_state.plan_generate_button_text = "Regenerate"

    if st.session_state.action_plan:
        st.subheader("Action Plan:")
        with st.form("planner_review"):
            action_plan = st.text_area(
                "Review and modify the action plan (Optional)",
                value=st.session_state.action_plan,
                height=400,
            )
            execute_actions = st.form_submit_button("Execute Actions")

            if execute_actions:
                st.session_state.action_plan = action_plan
                st.session_state.state["action_plan"] = action_plan
                st.session_state.step = Step.execute
                st.session_state.action_log.append(st.session_state.action_plan)
                with st.spinner("Executing", show_time=True):
                    state_update = graph.nodes["executor"].invoke(
                        st.session_state.state
                    )
                st.session_state.show_execute_section = True
                st.session_state.state.update(state_update.update)
                st.success("Done!")

    # Executor Step
    if st.session_state.show_execute_section and st.session_state.action_plan:
        st.subheader("⚙️ New Network Status")

        st.write(st.session_state.state["action_plan"])

        st.session_state.show_critic_section = True

        st.session_state.step = Step.critic

        net = pp.from_json(WORKSPACE_NETWORKS / "network.json")

        pp.runpp(net)

        st.session_state.line_violations_after_action = net.res_line[
            net.res_line.loading_percent > 100
        ]
        st.session_state.voltage_violations_after_action = net.res_bus[
            (net.res_bus.vm_pu > 1.05) | (net.res_bus.vm_pu < 0.95)
        ]

        if st.session_state.line_violations_after_action is not None:
            st.write("After Action Thermal Violations (if any):")
            st.dataframe(st.session_state.line_violations_after_action)

        if st.session_state.voltage_violations_after_action is not None:
            st.write("After Action Voltage Violations (if any):")
            st.dataframe(st.session_state.voltage_violations_after_action)

        with st.expander("Network Plot (click to expand)"):
            fig = pp.plotting.plotly.pf_res_plotly(net, auto_open=False)
            st.plotly_chart(fig, use_container_width=False)

    # Critic Step
    if st.session_state.show_critic_section:
        st.subheader("🧠 Critic Feedback")

        if st.button("Judge Action Plan"):
            with st.spinner("Judging action plan...", show_time=True):
                state_update = graph.nodes["critic"].invoke(st.session_state.state)

            st.session_state.state.update(state_update.update)
            st.session_state.feedback = state_update.update["feedback"]

            with st.container():
                st.write(f"Critic Feedback: {st.session_state.feedback}")

            if (
                len(st.session_state.state["violation_after_action"]["voltage"]) > 0
                or len(st.session_state.state["violation_after_action"]["thermal"]) > 0
            ):
                if st.button("Generate New Plan"):
                    st.session_state.step = Step.plan
                    st.session_state.iter += 1
            else:
                st.session_state.step = Step.done
                st.write("No violations found!")
