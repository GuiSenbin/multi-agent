from operator import add
from typing import TypedDict, Annotated
import asyncio

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_community.chat_models import ChatTongyi
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from config.load_key import load_key

nodes = ["supervisor", "travel", "couplet", "joke", "other"]

import os

os.environ["DASHSCOPE_API_KEY"] = load_key("BAILIAN_API_KEY")

llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=load_key("BAILIAN_API_KEY")
)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    type: str


def supervisor_node(state: State):
    print(">>> supervisor_node")
    writer = get_stream_writer()
    writer({"node", ">>>> supervisor_node"})
    
    prompt = """你是一个专业的客服助手，负责对用户的问题进行分类，并将任务分给其他Agent执行。
            如果用户的问题是和旅游路线规划相关的，那就返回 travel 。
            如果用户的问题是希望讲一个笑话，那就返回 joke 。
            如果用户的问题是希望对一个对联，那就返回 couplet 。
            如果是其他的问题，返回 other 。
            除了这几个选项外，不要返回任何其他的内容。
"""

    prompts = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": state["messages"][0]}
    ]
    
    # 如果已经有type属性了，表示问题已经交由其他节点处理完成了，就可以直接返回
    if "type" in state:
        writer({"supervisor_step": f"已获得{state['type']} 智能体处理结果"})
        return {"type": END}
    else:
        response = llm.invoke(prompts)
        typeRes = response.content.strip()
        writer({"supervisor_step": f"问题分类结果: {typeRes}"})
        if typeRes in nodes:
            return {"type": typeRes}
        else:
            raise ValueError("type is not in (travel,joke,other,couplet)")


def travel_node(state: State):
    print(">>> travel_node")
    writer = get_stream_writer()
    writer({"node": ">>>> travel_node"})
    
    system_prompt = "你是一个专业的旅行规划助手，根据用户的问题，生成一个旅游路线规划。请用中文回答，并返回一个不超过100字的规划结果"
    
    prompts = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["messages"][0]}
    ]
    
    # 高德地图的MCP配置信息 (使用远程 SSE 连接，无需本地安装 Node.js/npx)
    client = MultiServerMCPClient(
        {
            "amap-maps": {
                "transport": "streamable_http",
                "url": "https://mcp.amap.com/mcp?key=" + load_key("AMAP_API_KEY")
            }
        }
    )
    
    tools = asyncio.run(client.get_tools())
    
    # 仅通过以下几行将异步 MCP 工具转为同步工具，以解决 NotImplementedError 并保证能以同步方式运行
    from langchain_core.tools import StructuredTool
    sync_tools = []
    for t in tools:
        def create_sync_func(tool_obj):
            return lambda *args, **kwargs: asyncio.run(tool_obj.ainvoke(kwargs if kwargs else (args[0] if args else {})))
        sync_tools.append(StructuredTool(
            name=t.name,
            description=t.description,
            func=create_sync_func(t),
            coroutine=t._arun,
            args_schema=t.args_schema
        ))
    
    agent = create_react_agent(
        model=llm,
        tools=sync_tools
    )
    
    response = agent.invoke({"messages": prompts})
    writer({"travel_result": response["messages"][-1].content})
    
    return {"messages": [HumanMessage(content=response["messages"][-1].content)], "type": "travel"}


def joke_node(state: State):
    print(">>> joke_node")
    writer = get_stream_writer()
    writer({"node": ">>>> joke_node"})
    
    system_prompt = "你是一个笑话大师，根据用户的问题，写一个不超过100个字的笑话。"
    
    prompts=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["messages"][0]}
    ]
    response = llm.invoke(prompts)
    writer({"joke_result": response.content})
    
    return {"messages": [HumanMessage(content=response.content)], "type": "joke"}


def retrieve_similar_couplets(query_text: str, k: int = 10) -> list[str]:

    """从 Redis 向量数据库中检索最相似的对联样本。
    如果 Redis 未运行或使用的是不支持 FT._LIST 的标准版 Redis，则自动降级在本地内存计算向量相似度。
    """
    from langchain_community.embeddings import DashScopeEmbeddings
    import numpy as np
    
    embedding_model = DashScopeEmbeddings(model="text-embedding-v1")
    redis_url = "redis://localhost:6379"
    samples = []
    use_redis = False
    
    try:
        import redis
        from langchain_redis import RedisConfig, RedisVectorStore
        redis_client = redis.from_url(redis_url)
        if redis_client.ping():
            config = RedisConfig(
                index_name="couplet",
                redis_url=redis_url
            )
            vector_store = RedisVectorStore(embedding_model, config=config)
            use_redis = True
    except Exception as e:
        if "unknown command" in str(e) or "FT._LIST" in str(e):
            print(f"[Warning] Redis running but missing RediSearch modules (standard Redis). Falling back to in-memory similarity search...")
        else:
            print(f"[Warning] Redis not running: {e}. Falling back to in-memory similarity search...")
            
    if use_redis:
        try:
            scored_results = vector_store.similarity_search_with_score(query_text, k=k)
            return [doc.page_content for doc, score in scored_results]
        except Exception as e:
            print(f"Error querying Redis, falling back to in-memory: {e}")
            
    # 内存计算降级方案
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(current_dir, "../resource/couplettest.csv"))
    lines = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as file:
            for line in file:
                line_str = line.strip()
                if line_str:
                    lines.append(line_str)
                    
    if lines:
        query_emb = embedding_model.embed_query(query_text)
        doc_embs = embedding_model.embed_documents(lines)
        
        q_vec = np.array(query_emb)
        similarities = []
        for doc, emb in zip(lines, doc_embs):
            d_vec = np.array(emb)
            sim = np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec))
            similarities.append((doc, sim))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in similarities[:k]]
        
    return []


def couplet_node(state: State):
    print(">>> couplet_node")
    writer = get_stream_writer()
    writer({"node": ">>>> couplet_node"})
    
    # 1. 提取用户输入的上联
    user_query = state["messages"][0]
    query_text = user_query.content if hasattr(user_query, "content") else str(user_query)
        
    # 2. 从向量库（或内存计算）中检索最匹配的 10 条参考对联
    samples = retrieve_similar_couplets(query_text, k=10)
    
    # 3. 构建提示词，调用 Qwen 生成下联
    from langchain_core.prompts import ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的对联大师，你的任务是根据用户给出的上联，设计一个下联。
            回答时，可以参考下面的参考对联。
            参考对联：
            {samples}
            请用中文回答问题
            """),
        ("user", "{text}")
    ])
    
    prompt = prompt_template.invoke({"samples": samples, "text": query_text})
    response = llm.invoke(prompt)
    
    writer({"couplet_result": response.content})
    return {"messages": [HumanMessage(content=response.content)], "type": "couplet"}


def other_node(state: State):
    print(">>> other_node")
    writer = get_stream_writer()
    writer({"node": ">>>> other_node"})
    return {"messages": [HumanMessage(content="我暂时无法回答这个问题")], "type": "other"}


# 条件路由
def routing_func(state: State):
    if state["type"] == "travel":
        return "travel_node"
    elif state["type"] == "joke":
        return "joke_node"
    elif state["type"] == "couplet":
        return "couplet_node"
    elif state["type"] == END:
        return END
    else:
        return "other_node"


# 构建图
builder = StateGraph(State)
# 添加节点
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("travel_node", travel_node)
builder.add_node("joke_node", joke_node)
builder.add_node("couplet_node", couplet_node)
builder.add_node("other_node", other_node)
# 添加Edge
builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges(
    "supervisor_node",
    routing_func,
    ["travel_node", "joke_node", "couplet_node", "other_node", END]
)
builder.add_edge("travel_node", "supervisor_node")
builder.add_edge("joke_node", "supervisor_node")
builder.add_edge("couplet_node", "supervisor_node")
builder.add_edge("other_node", "supervisor_node")

# 构建Graph
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer = checkpointer)

# 执行任务的测试代码
if __name__ == "__main__":
    config = {
        "configurable":{
            "thread_id": "1"
        }
    }

    for chunk in graph.stream({"messages":["给我写一个对联：上联：瑞雪兆丰年"]}
                              ,config
                              ,stream_mode="custom"):
        print(chunk)

