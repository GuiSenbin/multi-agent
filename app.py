"""提供 Gradio 本地交互页面。"""

import random
import gradio as gr
from graph.builder import graph
from services.ui_trace import TraceDisplay, apply_trace_event, format_runtime_error

def process_input(text):
    config = {
        "configurable": {
            "thread_id": random.randint(1, 1000)
        }
    }
    trace_display = TraceDisplay()
    final_answer = ""

    try:
        # 调用 LangGraph 多智能体图，并收集中间自定义事件
        for chunk in graph.stream({"messages": [text]}, config, stream_mode=["values", "custom"]):
            stream_mode, payload = chunk
            if stream_mode == "custom" and isinstance(payload, dict):
                trace_display = apply_trace_event(trace_display, payload)
            elif stream_mode == "values" and payload.get("messages"):
                latest_message = payload["messages"][-1]
                final_answer = latest_message.content if hasattr(latest_message, "content") else str(latest_message)
    except Exception as error:
        final_answer = format_runtime_error(error)

    return trace_display.intent, trace_display.rag_references, final_answer

# 构建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# LangGraph Multi-Agent 智能客服系统")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 问答输入\n可以问路线规划、对对联、讲笑话，快来试试吧！")
            inputs_text = gr.Textbox(label="问题*", placeholder="请输入你的问题", value="讲一个郭德纲的笑话")
            btn_start = gr.Button(value="发送问题", variant="primary")
        with gr.Column():
            output_intent = gr.Textbox(label="意图识别结果", interactive=False)
            output_rag = gr.Textbox(label="RAG Top 3 参考内容", lines=6, interactive=False)
            output_text = gr.Textbox(label="智能体回复结果", interactive=False)

    btn_start.click(process_input, inputs=[inputs_text], outputs=[output_intent, output_rag, output_text])

if __name__ == "__main__":
    # 启动本地网页界面
    demo.launch(server_name="127.0.0.1", server_port=7863)
