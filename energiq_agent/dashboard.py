# dashboard.py

import streamlit as st
import json
import pandapower as pp

st.set_page_config(layout="wide")
st.title("⚡ Power Grid Violation Resolver")

# Upload JSON
uploaded_file = st.file_uploader("Upload pandapower JSON network", type=["json"])


def run_langgraph_agent(net: str):
    from energiq_agent.agents.graph import get_workflow

    workflow = get_workflow()
    graph = workflow.compile()
    result = graph.invoke(
        {
            "network_file_path": net,
        }
    )
    return result


if uploaded_file:
    network_data = json.load(uploaded_file)
    net = pp.from_json_string(json.dumps(network_data))
    pp.to_json(net, "network.json")

    st.success("✅ Network loaded successfully.")

    # Show basic grid info
    st.subheader("🔍 Network Overview")
    st.write(
        f"🧵 Buses: {len(net.bus)}, 🔌 Lines: {len(net.line)}, 🔁 Switches: {len(net.switch)}"
    )

    # Option to show violations before running agent
    if st.button("Check Initial Violations"):
        pp.runpp(net, calculate_voltage_angles=True)
        st.write("Thermal Violations (if any):")
        st.dataframe(net.res_line[net.res_line.loading_percent > 100])
        st.write("Voltage Violations (if any):")
        st.dataframe(
            net.res_bus[(net.res_bus.vm_pu > 1.05) | (net.res_bus.vm_pu < 0.95)]
        )

    # Run LangGraph planner-agent-critic loop
    if st.button("Run LangGraph Agent to Fix Violations"):
        with st.spinner("🤖 Agent working..."):
            result_state = run_langgraph_agent("network.json")
            st.success("Agent completed!")

        # Show result summary
        st.subheader("✅ Updated Network State")
        st.write(result_state["log"])
