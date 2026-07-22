"""构建 LangGraph 多智能体执行图。"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.couplet import couplet_node
from agents.joke import joke_node
from agents.other import other_node
from agents.supervisor import supervisor_node
from agents.travel import travel_node
from graph.state import State


def routing_func(state: State):
    if state["type"] == "travel":
        return "travel_node"
    if state["type"] == "joke":
        return "joke_node"
    if state["type"] == "couplet":
        return "couplet_node"
    if state["type"] == END:
        return END
    return "other_node"


def build_graph():
    builder = StateGraph(State)
    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node("travel_node", travel_node)
    builder.add_node("joke_node", joke_node)
    builder.add_node("couplet_node", couplet_node)
    builder.add_node("other_node", other_node)
    builder.add_edge(START, "supervisor_node")
    builder.add_conditional_edges(
        "supervisor_node",
        routing_func,
        ["travel_node", "joke_node", "couplet_node", "other_node", END],
    )
    builder.add_edge("travel_node", "supervisor_node")
    builder.add_edge("joke_node", "supervisor_node")
    builder.add_edge("couplet_node", "supervisor_node")
    builder.add_edge("other_node", "supervisor_node")

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
