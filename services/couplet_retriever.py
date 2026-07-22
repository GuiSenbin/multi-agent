"""提供对联样本的向量检索和本地降级检索。"""

from langchain_community.embeddings import DashScopeEmbeddings
import numpy as np

from config.model_config import EMBEDDING_MODEL_NAME
from config.paths import resource_path


def retrieve_similar_couplets(query_text: str, k: int = 10) -> list[str]:
    embedding_model = DashScopeEmbeddings(model=EMBEDDING_MODEL_NAME)
    redis_url = "redis://localhost:6379"
    use_redis = False

    try:
        import redis
        from langchain_redis import RedisConfig, RedisVectorStore

        redis_client = redis.from_url(redis_url)
        if redis_client.ping():
            config = RedisConfig(
                index_name="couplet",
                redis_url=redis_url,
            )
            vector_store = RedisVectorStore(embedding_model, config=config)
            use_redis = True
    except Exception as e:
        if "unknown command" in str(e) or "FT._LIST" in str(e):
            print("[Warning] Redis running but missing RediSearch modules (standard Redis). Falling back to in-memory similarity search...")
        else:
            print(f"[Warning] Redis not running: {e}. Falling back to in-memory similarity search...")

    if use_redis:
        try:
            scored_results = vector_store.similarity_search_with_score(query_text, k=k)
            return [doc.page_content for doc, score in scored_results]
        except Exception as e:
            print(f"Error querying Redis, falling back to in-memory: {e}")

    return _retrieve_from_memory(query_text, embedding_model, k)


def _retrieve_from_memory(query_text: str, embedding_model: DashScopeEmbeddings, k: int) -> list[str]:
    csv_path = resource_path("couplettest.csv")
    lines = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as file:
            for line in file:
                line_str = line.strip()
                if line_str:
                    lines.append(line_str)

    if not lines:
        return []

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
