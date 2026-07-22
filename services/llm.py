"""初始化项目使用的通义千问模型。"""

import os

from langchain_community.chat_models import ChatTongyi

from config.load_key import load_key
from config.model_config import CHAT_MODEL_NAME


API_KEY = load_key("BAILIAN_API_KEY")

if not os.environ.get("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = API_KEY

llm = ChatTongyi(
    model=CHAT_MODEL_NAME,
    dashscope_api_key=API_KEY,
)
