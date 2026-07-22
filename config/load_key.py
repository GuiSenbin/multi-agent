"""读取本地私密 Key 配置。"""

import json
import os

def load_key(key_name: str) -> str:
    # Load from Keys.json in the same directory as load_key.py
    config_dir = os.path.dirname(os.path.abspath(__file__))
    keys_path = os.path.join(config_dir, "Keys.json")
    with open(keys_path, "r", encoding="utf-8") as f:
        keys = json.load(f)
    return keys.get(key_name, "")
