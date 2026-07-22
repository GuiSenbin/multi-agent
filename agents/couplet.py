"""定义对联生成 Agent 节点。"""

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.config import get_stream_writer

from graph.state import State
from services.couplet_retriever import retrieve_similar_couplets
from services.llm import llm


def couplet_node(state: State):
    print(">>> couplet_node")
    writer = get_stream_writer()
    writer({"node": ">>>> couplet_node"})

    user_query = state["messages"][0]
    query_text = user_query.content if hasattr(user_query, "content") else str(user_query)

    samples = retrieve_similar_couplets(query_text, k=10)
    writer({"rag_references": samples[:3]})

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的对联大师，你的任务是根据用户给出的上联，设计一个下联。
            回答时，可以参考下面的参考对联。
            参考对联：
            {samples}
            请用中文回答问题
            """),
        ("user", "{text}"),
    ])

    prompt = prompt_template.invoke({"samples": samples, "text": query_text})
    response = llm.invoke(prompt)

    writer({"couplet_result": response.content})
    return {"messages": [HumanMessage(content=response.content)], "type": "couplet"}
