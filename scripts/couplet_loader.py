"""将对联文本加载到 Redis 向量数据库。"""

import os

import redis
from langchain_community.embeddings import DashScopeEmbeddings

from config.load_key import load_key
from config.model_config import EMBEDDING_MODEL_NAME
from config.paths import resource_path


if not os.environ.get("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = load_key("BAILIAN_API_KEY")

embedding_model = DashScopeEmbeddings(model=EMBEDDING_MODEL_NAME)
redis_url = "redis://localhost:6379"

try:
    redis_client = redis.from_url(redis_url)
    print("Testing connection to Redis...")
    if redis_client.ping():
        print("Redis connection successful!")
        from langchain_redis import RedisConfig, RedisVectorStore

        config = RedisConfig(
            index_name="couplet",
            redis_url=redis_url,
        )
        vector_store = RedisVectorStore(embedding_model, config=config)

        lines = []
        with resource_path("couplettest.csv").open("r", encoding="utf-8") as file:
            for line in file:
                line_str = line.strip()
                if line_str:
                    print(line_str)
                    lines.append(line_str)

        vector_store.add_texts(lines)
        print("Saved successfully to Redis vector database.")
except Exception as e:
    print(f"\n[Warning] Redis error: {e}")
    if "unknown command" in str(e) or "FT._LIST" in str(e):
        print("\n友情提示：这是因为你使用的是标准版的 Redis，而 langchain_redis 库运行向量数据库需要加载 RediSearch 搜索模块（即 Redis Stack）。")
        print("不用担心！无需重新安装，项目的降级机制会检测到这一点，并在运行主程序时自动切换为内存向量计算。")
    else:
        print("请检查你的 Redis 是否已启动。你可以通过以下命令在 Mac 上安装并启动 Redis：")
        print("  brew install redis")
        print("  brew services start redis")
    print("\n你可以直接运行 python3 app.py，主程序会自动在内存中进行相似度检索，不影响功能使用。")
