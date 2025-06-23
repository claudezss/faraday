from enum import Enum

import streamlit as st
import json
import pandas as pd
import pandapower as pp
from langgraph.types import Command
from dataclasses import dataclass, field

from energiq_agent.agents.graph import get_workflow, State

workflow = get_workflow()
graph = workflow.compile()


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

    #
    show_plan_section: bool = False


def initialize(session_state):
    if not session_state:
        return SessionState()
    else:
        return session_state


st.session_state = initialize(st.session_state)

show_state = st.sidebar.popover("Show State")
if show_state:
    show_state.write(st.session_state)

st.set_page_config(layout="wide")
st.title("⚡ Interactive Power Grid Agent")

# Upload grid JSON
uploaded_file = st.file_uploader("Upload pandapower JSON network", type=["json"])

if uploaded_file and st.session_state.net is None:
    network_data = json.load(uploaded_file)
    net = pp.from_json_string(json.dumps(network_data))
    pp.to_json(net, "network.json")
    st.session_state.net = net

# State: Network Loaded
if st.session_state.net is not None:
    st.subheader("📊 Network Overview")
    net = st.session_state.net
    st.write(
        f"🧵 Buses: {len(net.bus)}, 🔌 Lines: {len(net.line)}, 🔁 Switches: {len(net.switch)}"
    )

    if (
        st.session_state.line_violations is None
        or st.session_state.voltage_violations is None
    ):
        check_viol = st.button("Check Initial Violations")
        if check_viol:
            pp.runpp(net, calculate_voltage_angles=True)
            st.session_state.line_violations = net.res_line[
                net.res_line.loading_percent > 100
            ]
            st.session_state.voltage_violations = net.res_bus[
                (net.res_bus.vm_pu > 1.05) | (net.res_bus.vm_pu < 0.95)
            ]

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
            reason_the_violation_occurred="",
            editing_network_file_path="network.json",
            messages=[],
            network=net,
            work_dir=".",
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
                st.session_state.state["reason_the_violation_occurred"] = helper
                st.session_state.state["feedback"] = critic_feedback

                with st.spinner("Generating action plan", show_time=True):
                    state_update: Command = graph.nodes["planner"].invoke(
                        st.session_state.state
                    )
                st.session_state.action_plan = state_update.update["action_plan"]
                st.session_state.state.update(state_update.update)
                st.session_state.plan_generate_button_text = "Regenerate"

                st.rerun()

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
                st.session_state.state["action_plan"] = st.session_state.action_plan
                st.session_state.step = Step.execute
                st.session_state.action_log.append(st.session_state.action_plan)
                st.rerun()

    # Executor Step
    if st.session_state.step == Step.execute and st.session_state.action_plan:
        st.subheader("⚙️ Executing Plan")
        with st.spinner("Executing", show_time=True):
            state_update = graph.nodes["executor"].invoke(st.session_state.state)
        st.success("Done!")

        st.session_state.state.update(state_update.update)
        st.session_state.step = Step.critic

        net = st.session_state.state["network"]

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

    # Critic Step
    if st.session_state.step == Step.critic:
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

            st.rerun()
