import random
import gradio as gr
from Director import graph

def process_input(text):
    config = {
        "configurable": {
            "thread_id": random.randint(1, 1000)
        }
    }
    # 调用我们的 LangGraph 多智能体图
    result = graph.invoke({"messages": [text]}, config)
    return result["messages"][-1].content

# 构建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# LangGraph Multi-Agent 智能客服系统")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 问答输入\n可以问路线规划、对对联、讲笑话，快来试试吧！")
            inputs_text = gr.Textbox(label="问题*", placeholder="请输入你的问题", value="讲一个郭德纲的笑话")
            btn_start = gr.Button(value="发送问题", variant="primary")
        with gr.Column():
            output_text = gr.Textbox(label="智能体回复结果", interactive=False)

    btn_start.click(process_input, inputs=[inputs_text], outputs=[output_text])

if __name__ == "__main__":
    # 启动本地网页界面
    demo.launch(server_name="127.0.0.1", server_port=7863)
