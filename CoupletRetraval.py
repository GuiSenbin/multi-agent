import os
from config.load_key import load_key
import redis
import numpy as np
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

query = "帮我对个对联，上联是：瑞雪兆丰年"


if not os.environ.get("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = load_key("BAILIAN_API_KEY")

embedding_model = DashScopeEmbeddings(model="text-embedding-v1")

redis_url = "redis://localhost:6379"
samples = []
use_redis = False

try:
    # Try connecting to Redis first
    redis_client = redis.from_url(redis_url)
    if redis_client.ping():
        from langchain_redis import RedisConfig, RedisVectorStore
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
        print(f"[Warning] Redis connection failed: {e}. Falling back to in-memory similarity search...")


if use_redis:
    try:
        scored_results = vector_store.similarity_search_with_score(query, k=10)
        for doc, score in scored_results:
            samples.append(doc.page_content)
    except Exception as e:
        print(f"Error querying Redis, falling back to in-memory: {e}")
        use_redis = False

if not use_redis:
    # In-memory fallback: read CSV, embed, and calculate cosine similarity
    csv_path = "../resource/couplettest.csv"
    lines = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as file:
            for line in file:
                line_str = line.strip()
                if line_str:
                    lines.append(line_str)
    
    if lines:
        print(f"Loaded {len(lines)} reference couplets from CSV for in-memory matching...")
        query_emb = embedding_model.embed_query(query)
        doc_embs = embedding_model.embed_documents(lines)
        
        q_vec = np.array(query_emb)
        similarities = []
        for doc, emb in zip(lines, doc_embs):
            d_vec = np.array(emb)
            sim = np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec))
            similarities.append((doc, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        for doc, score in similarities[:10]:
            samples.append(doc)
    else:
        print("[Warning] No reference couplets found in CSV!")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的对联大师，你的任务是根据用户给出的上联，设计一个下联。
            回答时，可以参考下面的参考对联。
            参考对联：
            {samples}
            请用中文回答问题
            """),
    ("user", "{text}")
])

prompt = prompt_template.invoke({"samples": samples, "text": query})

print("\n--- Generated Prompt ---")
print(prompt)

llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=load_key("BAILIAN_API_KEY")
)

response = llm.invoke(prompt)
print("\n---- 回复结果 ----")
print(response.content)
