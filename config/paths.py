"""提供项目资源文件的稳定路径解析。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resource_path(*parts: str) -> Path:
    return PROJECT_ROOT / "resource" / Path(*parts)
