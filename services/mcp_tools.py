"""适配高德 MCP 异步工具为同步工具。"""

import asyncio

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from config.load_key import load_key


def get_amap_sync_tools():
    client = MultiServerMCPClient(
        {
            "amap-maps": {
                "transport": "streamable_http",
                "url": "https://mcp.amap.com/mcp?key=" + load_key("AMAP_API_KEY"),
            }
        }
    )

    tools = asyncio.run(client.get_tools())
    sync_tools = []
    for tool in tools:
        sync_tools.append(StructuredTool(
            name=tool.name,
            description=tool.description,
            func=_create_sync_func(tool),
            coroutine=tool._arun,
            args_schema=tool.args_schema,
        ))
    return sync_tools


def _create_sync_func(tool):
    return lambda *args, **kwargs: asyncio.run(tool.ainvoke(kwargs if kwargs else (args[0] if args else {})))
