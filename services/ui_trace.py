"""整理智能体执行事件的界面展示状态。"""

from dataclasses import dataclass
from typing import Any

from config.model_config import CHAT_MODEL_NAME


@dataclass
class TraceDisplay:
    intent: str = "等待识别"
    rag_references: str = "非 RAG 检索流程，暂无参考内容"


def apply_trace_event(display: TraceDisplay, event: dict[str, Any]) -> TraceDisplay:
    if "supervisor_step" in event:
        step = str(event["supervisor_step"])
        marker = "问题分类结果:"
        if marker in step:
            display.intent = step.split(marker, 1)[1].strip()

    if "rag_references" in event:
        references = [str(item) for item in event["rag_references"][:3]]
        display.rag_references = "\n".join(
            f"{index}. {reference}" for index, reference in enumerate(references, start=1)
        )
        if not display.rag_references:
            display.rag_references = "未检索到相似参考"

    return display


def format_runtime_error(error: Exception) -> str:
    message = str(error)
    if "AllocationQuota.FreeTierOnly" in message or "Free quota exhausted" in message:
        return f"百炼接口返回 FreeTierOnly：请确认项目使用的 API Key、业务空间、模型名和控制台额度页面一致；当前项目已配置为 {CHAT_MODEL_NAME}。"
    return f"运行出错：{message}"
