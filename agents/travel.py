"""定义旅游规划 Agent 节点。"""

from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer
from langgraph.prebuilt import create_react_agent

from graph.state import State
from services.llm import llm
from services.mcp_tools import get_amap_sync_tools


def travel_node(state: State):
    print(">>> travel_node")
    writer = get_stream_writer()
    writer({"node": ">>>> travel_node"})

    system_prompt = "你是一个专业的旅行规划助手，根据用户的问题，生成一个旅游路线规划。请用中文回答，并返回一个不超过100字的规划结果"

    prompts = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["messages"][0]},
    ]

    agent = create_react_agent(
        model=llm,
        tools=get_amap_sync_tools(),
    )

    response = agent.invoke({"messages": prompts})
    writer({"travel_result": response["messages"][-1].content})

    return {"messages": [HumanMessage(content=response["messages"][-1].content)], "type": "travel"}
