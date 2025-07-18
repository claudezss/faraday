"""
Explainer node for generating explanations and collecting training data.
"""

import json

from langgraph.types import Command

from faraday import WORKSPACE
from faraday.agents.planner import Planner
from faraday.agents.prompts import EXPLAINER_PROMPT
from faraday.agents.workflow.state import State
from faraday.tools.pandapower import read_network, get_network_status
from ..config import llm


def explainer(state: State):
    """Generates an explanation of the actions and saves the conversation data."""
    executed_actions = state.get("executed_actions", [])
    if not executed_actions:
        explanation = "No actions were executed, so no explanation is needed."
    else:
        action_report = "\n".join(
            [f"- {action['name']}({action['args']})" for action in executed_actions]
        )
        explanation_prompt = EXPLAINER_PROMPT.format(
            violation_before_action=state["violation_before_action"],
            action_report=action_report,
            violation_after_action=state["violation_after_action"],
        )
        explanation = llm.invoke(explanation_prompt).content

    # --- Enhanced Data Collection for Fine-tuning ---
    try:
        from faraday.training.data_collector import EnhancedTrainingDataCollector

        # Initialize enhanced collector
        training_dir = WORKSPACE / "training_data_enhanced"
        collector = EnhancedTrainingDataCollector(training_dir)

        # Collect comprehensive training sample
        training_sample = collector.collect_training_sample(
            state=state,
            executed_actions=executed_actions,
            explanation=explanation,
            system_prompt=Planner.prompt(),
        )

        # Save in multiple formats
        collector.save_training_sample(training_sample)

    except Exception as e:
        # Fallback to legacy method if enhanced collection fails
        print(f"Enhanced training data collection failed: {e}")
        _fallback_training_data_collection(state, executed_actions, explanation)

    return Command(update={"explanation": explanation})


def _fallback_training_data_collection(state, executed_actions, explanation):
    """Fallback method for training data collection."""
    training_data_path = WORKSPACE / "training_data.json"

    net = read_network(state["network_file_path"])
    status = get_network_status(net)
    conversation_data = {
        "system_prompt": Planner.prompt(),
        "user_prompt": f"""Network Status: \n{status}\n""",
        "assistant_response": {
            "actions": executed_actions,
            "explanation": explanation,
        },
    }

    # Append the new data to the JSON file
    if training_data_path.exists():
        with open(training_data_path, "r+") as f:
            data = json.load(f)
            data.append(conversation_data)
            f.seek(0)
            json.dump(data, f, indent=2)
    else:
        with open(training_data_path, "w") as f:
            json.dump([conversation_data], f, indent=2)
