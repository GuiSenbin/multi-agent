"""定义主管 Agent 的分类与收束逻辑。"""

from langgraph.config import get_stream_writer
from langgraph.graph import END

from graph.state import State
from services.llm import llm


NODES = ["supervisor", "travel", "couplet", "joke", "other"]


def supervisor_node(state: State):
    print(">>> supervisor_node")
    writer = get_stream_writer()
    writer({"node": ">>>> supervisor_node"})

    prompt = """你是一个专业的客服助手，负责对用户的问题进行分类，并将任务分给其他Agent执行。
            如果用户的问题是和旅游路线规划相关的，那就返回 travel 。
            如果用户的问题是希望讲一个笑话，那就返回 joke 。
            如果用户的问题是希望对一个对联，那就返回 couplet 。
            如果是其他的问题，返回 other 。
            除了这几个选项外，不要返回任何其他的内容。
"""

    prompts = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": state["messages"][0]},
    ]

    if "type" in state:
        writer({"supervisor_step": f"已获得{state['type']} 智能体处理结果"})
        return {"type": END}

    response = llm.invoke(prompts)
    type_res = response.content.strip()
    writer({"supervisor_step": f"问题分类结果: {type_res}"})
    if type_res in NODES:
        return {"type": type_res}
    raise ValueError("type is not in (travel,joke,other,couplet)")
