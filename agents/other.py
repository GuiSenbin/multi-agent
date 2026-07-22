"""定义无法处理问题的兜底 Agent 节点。"""

from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer

from graph.state import State


def other_node(state: State):
    print(">>> other_node")
    writer = get_stream_writer()
    writer({"node": ">>>> other_node"})
    return {"messages": [HumanMessage(content="我暂时无法回答这个问题")], "type": "other"}
