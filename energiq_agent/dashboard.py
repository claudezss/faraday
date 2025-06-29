import json
from enum import Enum
import pandas as pd
import pandapower as pp
import streamlit as st
from dataclasses import dataclass, field

from energiq_agent import ROOT_DIR
from energiq_agent.agents.graph import (
    planner as run_planner,
    executor as run_executor,
    summarizer as run_summarizer,
    cache_network,
)
from energiq_agent.schemas import State

# --- Initialization ---
WORKSPACE = ROOT_DIR.parent / "workspace"
WORKSPACE_NETWORKS = WORKSPACE / "networks"
WORKSPACE_NETWORKS.mkdir(parents=True, exist_ok=True)
MAX_ITER = 5


class Step(str, Enum):
    START = "start"
    PLAN_GENERATED = "plan_generated"
    EXECUTED = "executed"


@dataclass
class ChatSessionState:
    step: Step = Step.START
    messages: list = field(default_factory=list)
    state: State = field(default_factory=dict)
    net: pp.pandapowerNet = None
    initial_line_violations: pd.DataFrame = field(default_factory=pd.DataFrame)
    initial_voltage_violations: pd.DataFrame = field(default_factory=pd.DataFrame)


def initialize_session_state():
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = ChatSessionState()


def valid_json(json_str):
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError as e:
        print(e)
        return False


initialize_session_state()
chat_state = st.session_state.chat_state

# --- UI Components ---
st.set_page_config(layout="wide")
st.title("⚡ Interactive Power Grid Agent Chat")


# --- Helper Functions ---
def get_violations(net):
    pp.runpp(net)
    line_violations = net.res_line[net.res_line.loading_percent > 100]
    voltage_violations = net.res_bus[
        (net.res_bus.vm_pu > 1.05) | (net.res_bus.vm_pu < 0.95)
    ]
    return line_violations, voltage_violations


def display_violation_data(container, line_violations, voltage_violations):
    container.write("Thermal Violations:")
    container.dataframe(line_violations)
    container.write("Voltage Violations:")
    container.dataframe(voltage_violations)


def add_message(role, content, plan_df=None):
    chat_state.messages.append({"role": role, "content": content, "plan_df": plan_df})


def format_plan_as_table(plan):
    if not plan:
        return None
    table_data = []
    for action in plan:
        tool_name = action.get("name", "Unknown Action")
        args = action.get("args", {})
        element = args.get("switch_name", args.get("load_name", args.get("bus_index")))
        params = {
            k: v for k, v in args.items() if k not in ["switch_name", "load_name"]
        }
        table_data.append(
            {
                "Action": tool_name,
                "Element": element,
                "Parameters": json.dumps(params),
            }
        )
    return pd.DataFrame(table_data)


def get_assistant_response(user_input: str):
    add_message("user", user_input)

    # --- Main Logic ---
    if chat_state.step == Step.START:
        with st.spinner("Analyzing network and generating a plan..."):
            cache_command = cache_network(chat_state.state)
            chat_state.state.update(cache_command.update)
            planner_command = run_planner(chat_state.state)
            chat_state.state.update(planner_command.update)
            action_plan = chat_state.state.get("action_plan")

            if not action_plan:
                add_message(
                    "assistant",
                    "No violations found or no action plan could be generated.",
                )
                chat_state.step = Step.START
            else:
                plan_df = format_plan_as_table(action_plan)
                json_plan_str = json.dumps(action_plan, indent=2)
                message = (
                    "I have analyzed the network and here is the proposed action plan. Please review it.\n\n"
                    "You can approve this plan by typing **'yes'**.\n\n"
                    "To modify the plan, copy the text block below, edit it, and paste it back into the chat."
                    f"\n\n```json\n{json_plan_str}\n```"
                )
                add_message("assistant", message, plan_df=plan_df)
                chat_state.step = Step.PLAN_GENERATED

    elif chat_state.step == Step.PLAN_GENERATED:
        approved = False
        if user_input.strip().lower() in ["yes", "y", "approve"]:
            approved = True
        else:
            try:
                cleaned_input = user_input.strip()
                if valid_json(cleaned_input):
                    print(True)
                    cleaned_input = json.loads(cleaned_input)
                else:
                    print(False)
                # modified_plan = ast.literal_eval(cleaned_input)
                modified_plan = cleaned_input
                if isinstance(modified_plan, list):
                    approved = True
                    chat_state.state["action_plan"] = modified_plan
                    add_message(
                        "assistant", "Thank you. I will now execute the modified plan."
                    )
                else:
                    add_message(
                        "assistant",
                        "That doesn't look like a valid plan. A plan should be a list of actions. Please try again.",
                    )
            except (SyntaxError, ValueError):
                add_message(
                    "assistant",
                    "I didn't understand that. Please either approve the plan with 'yes' or provide a valid modified plan.",
                )

        if approved:
            with st.spinner("Executing the plan..."):
                executor_command = run_executor(chat_state.state)
                chat_state.state.update(executor_command.update)
                final_net = pp.from_json(chat_state.state["editing_network_file_path"])
                line_violations_after, voltage_violations_after = get_violations(
                    final_net
                )
                has_violations = not (
                    line_violations_after.empty and voltage_violations_after.empty
                )
                max_iter_reached = chat_state.state.get("iter", 0) >= MAX_ITER

                if has_violations and not max_iter_reached:
                    add_message(
                        "assistant",
                        "The plan was executed, but some violations remain. I will generate a new plan to address them.",
                    )
                    with st.spinner("Generating a new plan..."):
                        planner_command = run_planner(chat_state.state)
                        chat_state.state.update(planner_command.update)
                        action_plan = chat_state.state.get("action_plan")
                        if not action_plan:
                            add_message(
                                "assistant",
                                "I could not generate a new plan. The process will now stop.",
                            )
                            chat_state.step = Step.EXECUTED
                        else:
                            plan_df = format_plan_as_table(action_plan)
                            json_plan_str = json.dumps(action_plan, indent=2)
                            message = (
                                "Here is the new proposed plan. Please review it.\n\n"
                                "You can approve this plan by typing **'yes'**.\n\n"
                                "To modify the plan, copy the text block below, edit it, and paste it back into the chat."
                                f"\n\n```json\n{json_plan_str}\n```"
                            )
                            add_message("assistant", message, plan_df=plan_df)
                            chat_state.step = Step.PLAN_GENERATED
                else:
                    with st.spinner("Summarizing the actions..."):
                        summarizer_command = run_summarizer(chat_state.state)
                        chat_state.state.update(summarizer_command.update)
                        summary = chat_state.state.get("summary", "Execution complete.")
                        add_message("assistant", summary)
                        chat_state.step = Step.EXECUTED


# --- Main App Flow ---
uploaded_file = st.file_uploader("Upload pandapower JSON network", type=["json"])

if uploaded_file and chat_state.net is None:
    network_data = json.load(uploaded_file)
    net = pp.from_json_string(json.dumps(network_data))
    network_file_path = WORKSPACE_NETWORKS / "network.json"
    pp.to_json(net, str(network_file_path))
    chat_state.net = net
    chat_state.state = State(
        network_file_path=str(network_file_path),
        editing_network_file_path=str(network_file_path),
        work_dir=str(WORKSPACE_NETWORKS.absolute()),
        messages=[],
        network=net,
        iter=0,
    )
    # Store initial violations
    line_v, volt_v = get_violations(net)
    chat_state.initial_line_violations = line_v
    chat_state.initial_voltage_violations = volt_v

if chat_state.net is not None:
    # Display initial network state if no chat has started

    st.subheader("Initial Network Status")
    display_violation_data(
        st, chat_state.initial_line_violations, chat_state.initial_voltage_violations
    )
    with st.expander("Initial Network Plot"):
        fig = pp.plotting.plotly.pf_res_plotly(chat_state.net, auto_open=False)
        st.plotly_chart(fig, use_container_width=True)
    if not chat_state.messages:
        add_message(
            "assistant",
            "Hello! I have loaded the network. I am ready to help you resolve any power grid violations. What would you like to do? For example, you can ask me to `fix the violations`.",
        )

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for msg in chat_state.messages:
            with st.chat_message(msg["role"]):
                if msg.get("plan_df") is not None:
                    st.dataframe(msg["plan_df"], use_container_width=True)
                st.markdown(msg["content"], unsafe_allow_html=True)

    # After execution, show the side-by-side comparison
    if chat_state.step == Step.EXECUTED:
        final_net = pp.from_json(chat_state.state["editing_network_file_path"])
        final_line_violations, final_voltage_violations = get_violations(final_net)

        st.header("Comparison View")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Initial Network Status")
            display_violation_data(
                st,
                chat_state.initial_line_violations,
                chat_state.initial_voltage_violations,
            )
            with st.expander("Initial Network Plot", expanded=True):
                fig = pp.plotting.plotly.pf_res_plotly(chat_state.net, auto_open=False)
                st.plotly_chart(fig, use_container_width=True, key="initial_plot")
        with col2:
            st.subheader("Final Network Status")
            display_violation_data(st, final_line_violations, final_voltage_violations)
            with st.expander("Final Network Plot", expanded=True):
                fig = pp.plotting.plotly.pf_res_plotly(final_net, auto_open=False)
                st.plotly_chart(fig, use_container_width=True, key="final_plot")

        # Final summary message
        has_violations = not (
            final_line_violations.empty and final_voltage_violations.empty
        )
        if not has_violations:
            st.success("All violations resolved successfully!")
        elif chat_state.state.get("iter", 0) >= MAX_ITER:
            st.warning(
                f"Max iterations ({MAX_ITER}) reached. Could not resolve all violations."
            )
        else:
            st.warning("Some violations still remain.")

        # Reset for next interaction
        chat_state.step = Step.START
        chat_state.state["iter"] = 0

    # Chat input should be last
    if prompt := st.chat_input("What should I do?"):
        get_assistant_response(prompt)
        st.rerun()
