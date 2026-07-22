"""测试对联相似检索和下联生成流程。"""

from langchain_core.prompts import ChatPromptTemplate

from services.couplet_retriever import retrieve_similar_couplets
from services.llm import llm


query = "帮我对个对联，上联是：瑞雪兆丰年"
samples = retrieve_similar_couplets(query, k=10)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的对联大师，你的任务是根据用户给出的上联，设计一个下联。
            回答时，可以参考下面的参考对联。
            参考对联：
            {samples}
            请用中文回答问题
            """),
    ("user", "{text}"),
])

prompt = prompt_template.invoke({"samples": samples, "text": query})

print("\n--- Generated Prompt ---")
print(prompt)

response = llm.invoke(prompt)
print("\n---- 回复结果 ----")
print(response.content)
