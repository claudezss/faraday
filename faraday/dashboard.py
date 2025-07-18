import json
from enum import Enum
import pandas as pd
import pandapower as pp
import streamlit as st
from dataclasses import dataclass, field

from faraday import WORKSPACE_NETWORKS
from faraday.agents.graph import (
    planner as run_planner,
    executor as run_executor,
    summarizer as run_summarizer,
    explainer as run_explainer,
    cache_network,
    get_workflow,
)
from faraday.agents.workflow.state import State
from faraday.tools.pandapower import (
    get_voltage_thresholds,
    set_voltage_thresholds,
)

# --- Initialization ---

MAX_ITER = 5


class Step(str, Enum):
    START = "start"
    PLAN_GENERATED = "plan_generated"
    EXECUTED = "executed"
    AUTO_RUNNING = "auto_running"
    AUTO_COMPLETED = "auto_completed"


@dataclass
class ChatSessionState:
    step: Step = Step.START
    messages: list = field(default_factory=list)
    state: State = field(default_factory=dict)
    net: pp.pandapowerNet = None
    initial_line_violations: pd.DataFrame = field(default_factory=pd.DataFrame)
    initial_voltage_violations: pd.DataFrame = field(default_factory=pd.DataFrame)
    auto_mode: bool = False
    auto_result: dict = field(default_factory=dict)


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
    thresholds = get_voltage_thresholds()
    voltage_violations = net.res_bus[
        (net.res_bus.vm_pu > thresholds.v_max) | (net.res_bus.vm_pu < thresholds.v_min)
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


def run_automated_workflow():
    """Run the complete automated workflow without human intervention."""
    chat_state.step = Step.AUTO_RUNNING

    # Initialize progress tracking
    progress_container = st.empty()
    status_container = st.empty()

    try:
        # Get workflow
        workflow = get_workflow()
        graph = workflow.compile()

        # Run the complete workflow
        with st.spinner("Running automated violation fix workflow..."):
            progress_container.info("🔧 Starting automated workflow...")

            result = graph.invoke(chat_state.state)

            # Store results
            chat_state.auto_result = result
            chat_state.state.update(result)

            progress_container.success("✅ Automated workflow completed!")

            # Display results immediately
            executed_actions = result.get("executed_actions", [])
            final_violations = result.get(
                "violation_after_action", {"voltage": [], "thermal": []}
            )
            summary = result.get("summary", "")
            explanation = result.get("explanation", "")

            # Show executed actions
            if executed_actions:
                st.subheader("🔧 Executed Actions")
                actions_df = format_plan_as_table(executed_actions)
                if actions_df is not None:
                    st.dataframe(actions_df, use_container_width=True)

                actions_summary = (
                    f"Executed {len(executed_actions)} actions: "
                    + ", ".join(
                        [
                            f"{action.get('name', 'Unknown')}({list(action.get('args', {}).values())})"
                            for action in executed_actions[:3]
                        ]
                    )
                    + ("..." if len(executed_actions) > 3 else "")
                )
                add_message(
                    "assistant",
                    f"✅ Automated workflow completed!\n\n{actions_summary}",
                )
            else:
                add_message(
                    "assistant",
                    "ℹ️ No actions were needed - network was already compliant.",
                )

            # Show summary and explanation
            if summary:
                add_message("assistant", f"📝 **Summary**: {summary}")

            if explanation:
                add_message("assistant", f"💡 **Explanation**: {explanation}")

            # Final status message
            total_violations = len(final_violations.get("voltage", [])) + len(
                final_violations.get("thermal", [])
            )
            if total_violations == 0:
                status_container.success("🎉 All violations successfully resolved!")
                add_message("assistant", "🎉 All violations successfully resolved!")
            else:
                status_container.warning(
                    f"⚠️ {total_violations} violations remaining after workflow completion."
                )
                add_message(
                    "assistant",
                    f"⚠️ {total_violations} violations remaining after workflow completion.",
                )

            chat_state.step = Step.AUTO_COMPLETED

    except Exception as e:
        progress_container.error(f"❌ Error in automated workflow: {str(e)}")
        add_message("assistant", f"❌ Error in automated workflow: {str(e)}")
        chat_state.step = Step.START


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

                        explainer_command = run_explainer(chat_state.state)
                        chat_state.state.update(explainer_command.update)
                        explanation = chat_state.state.get("explanation", "")
                        add_message("assistant", explanation)

                        chat_state.step = Step.EXECUTED


# --- Voltage Threshold Configuration ---
st.sidebar.header("⚙️ Configuration")

# Voltage thresholds
st.sidebar.subheader("Voltage Thresholds")
current_thresholds = get_voltage_thresholds()

v_max = st.sidebar.number_input(
    "Maximum Voltage (p.u.)",
    min_value=1.00,
    max_value=1.20,
    value=current_thresholds.v_max,
    step=0.01,
    format="%.2f",
    help="Maximum allowed voltage in per unit",
)

v_min = st.sidebar.number_input(
    "Minimum Voltage (p.u.)",
    min_value=0.80,
    max_value=1.00,
    value=current_thresholds.v_min,
    step=0.01,
    format="%.2f",
    help="Minimum allowed voltage in per unit",
)

# Update thresholds if they changed
if v_max != current_thresholds.v_max or v_min != current_thresholds.v_min:
    set_voltage_thresholds(v_max=v_max, v_min=v_min)
    st.sidebar.success(f"✅ Updated thresholds: v_max={v_max}, v_min={v_min}")
    # If network is loaded, refresh violation detection
    if chat_state.net is not None:
        line_v, volt_v = get_violations(chat_state.net)
        chat_state.initial_line_violations = line_v
        chat_state.initial_voltage_violations = volt_v

st.sidebar.info(
    f"Current: v_max={current_thresholds.v_max}, v_min={current_thresholds.v_min}"
)

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
    # Mode selection
    st.subheader("Workflow Mode")
    col_mode_1, col_mode_2 = st.columns(2)

    with col_mode_1:
        if st.button(
            "🤖 Run Automated Fix",
            key="auto_mode_btn",
            help="Automatically fix all violations without human intervention",
            disabled=(
                chat_state.step
                in [Step.AUTO_RUNNING, Step.AUTO_COMPLETED, Step.PLAN_GENERATED]
            ),
            use_container_width=True,
        ):
            run_automated_workflow()
            st.rerun()

    with col_mode_2:
        if st.button(
            "💬 Interactive Mode",
            key="interactive_mode_btn",
            help="Step-by-step interactive mode with plan approval",
            disabled=(chat_state.step in [Step.AUTO_RUNNING, Step.AUTO_COMPLETED]),
            use_container_width=True,
        ):
            chat_state.step = Step.START
            chat_state.auto_mode = False
            if not chat_state.messages:
                add_message(
                    "assistant",
                    "Hello! I'm ready to help you resolve power grid violations in interactive mode. "
                    "You can ask me to `fix the violations` and I'll show you the plan for approval.",
                )
            st.rerun()

    # Display initial network state if no chat has started
    st.subheader("Initial Network Status")
    init_net_col_1, init_net_col_2 = st.columns(2)

    display_violation_data(
        init_net_col_1,
        chat_state.initial_line_violations,
        chat_state.initial_voltage_violations,
    )
    init_net_col_1.write("Load Info:")
    init_net_col_1.dataframe(chat_state.net.load)

    fig = pp.plotting.plotly.pf_res_plotly(chat_state.net, auto_open=False)
    init_net_col_2.write("Network Plot:")
    init_net_col_2.plotly_chart(fig, use_container_width=True)
    init_net_col_2.write("Generator Info:")
    init_net_col_2.dataframe(chat_state.net.sgen)

    if not chat_state.messages:
        add_message(
            "assistant",
            "Hello! I have loaded the network. I am ready to help you resolve any power grid violations. "
            "What would you like to do? For example, you can ask me to `fix the violations`",
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
    if chat_state.step in [Step.EXECUTED, Step.AUTO_COMPLETED]:
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

    # Chat input should be last (disabled during automated mode)
    chat_disabled = chat_state.step in [Step.AUTO_RUNNING, Step.AUTO_COMPLETED]
    placeholder_text = (
        "Automated mode active..." if chat_disabled else "What should I do?"
    )

    if prompt := st.chat_input(placeholder_text, disabled=chat_disabled):
        get_assistant_response(prompt)
        st.rerun()
