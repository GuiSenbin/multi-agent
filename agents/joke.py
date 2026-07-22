"""定义笑话生成 Agent 节点。"""

from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer

from graph.state import State
from services.llm import llm


def joke_node(state: State):
    print(">>> joke_node")
    writer = get_stream_writer()
    writer({"node": ">>>> joke_node"})

    system_prompt = "你是一个笑话大师，根据用户的问题，写一个不超过100个字的笑话。"

    prompts = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["messages"][0]},
    ]
    response = llm.invoke(prompts)
    writer({"joke_result": response.content})

    return {"messages": [HumanMessage(content=response.content)], "type": "joke"}
