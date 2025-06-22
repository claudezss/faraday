# dashboard.py

import streamlit as st
import json
import pandas as pd
import pandapower as pp


def mock_planner(net):
    return [{"action": "open_switch", "switch_index": 0}]


def mock_executor(net, action):
    try:
        net.switch.at[action["switch_index"], "closed"] = False
        return {"status": "success"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def mock_critic(net):
    pp.runpp(net)
    violations = net.res_line[net.res_line.loading_percent > 100]
    return {"violations": len(violations), "details": violations}


st.set_page_config(layout="wide")
st.title("⚡ Interactive Power Grid Agent")

# Upload grid JSON
uploaded_file = st.file_uploader("Upload pandapower JSON network", type=["json"])
if uploaded_file and "net" not in st.session_state:
    network_data = json.load(uploaded_file)
    net = pp.from_json_string(json.dumps(network_data))
    st.session_state.net = net
    st.session_state.step = "planner"
    st.session_state.action_log = []

# State: Network Loaded
if "net" in st.session_state:
    st.subheader("📊 Network Overview")
    net = st.session_state.net
    st.write(
        f"🧵 Buses: {len(net.bus)}, 🔌 Lines: {len(net.line)}, 🔁 Switches: {len(net.switch)}"
    )

    if st.button("Run Power Flow"):
        pp.runpp(net)
        st.dataframe(net.res_line[["loading_percent"]])

    # Planner Step
    if st.session_state.step == "planner":
        st.subheader("🧠 Planner Suggestions")
        st.session_state.action_plan = mock_planner(net)
        for i, action in enumerate(st.session_state.action_plan):
            with st.expander(f"Suggested Action #{i + 1}"):
                st.json(action)

        if st.button("✅ Approve Plan"):
            st.session_state.step = "executor"
        if st.button("✏️ Modify Plan"):
            idx = st.number_input(
                "Switch Index",
                value=st.session_state.action_plan[0]["switch_index"],
                min_value=0,
                max_value=len(net.switch) - 1,
            )
            st.session_state.action_plan[0]["switch_index"] = idx

    # Executor Step
    elif st.session_state.step == "executor":
        st.subheader("⚙️ Executing Plan")
        action = st.session_state.action_plan[0]
        result = mock_executor(net, action)
        st.session_state.action_log.append(
            {"step": "executor", "action": action, **result}
        )
        st.write(result)

        if result["status"] == "success":
            st.session_state.step = "critic"
        else:
            st.session_state.step = "planner"

    # Critic Step
    elif st.session_state.step == "critic":
        st.subheader("🧠 Critic Feedback")
        result = mock_critic(net)
        st.write(f"Violations remaining: {result['violations']}")
        st.dataframe(result["details"])

        st.session_state.action_log.append({"step": "critic", "result": result})

        if result["violations"] == 0:
            st.success("✅ All violations resolved.")
            st.session_state.step = "done"
        else:
            st.session_state.step = "planner"

    elif st.session_state.step == "done":
        st.success("🎉 Agent loop finished successfully.")

    # Action Log
    if st.session_state.action_log:
        st.subheader("📜 Agent Action Log")
        st.dataframe(pd.DataFrame(st.session_state.action_log))

    # Reset
    if st.button("🔁 Reset"):
        for key in ["net", "step", "action_plan", "action_log"]:
            st.session_state.pop(key, None)
        st.rerun()

    # Download
    st.download_button(
        "📥 Download Updated Network",
        data=pp.to_json(net),
        file_name="updated_net.json",
    )
