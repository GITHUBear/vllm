import h5py
import json
import numpy as np


def load_json_by_key(filepath, key):
    """
    根据 hash key 加载对应的 JSON 数据
    """
    with h5py.File(filepath, 'r') as f:
        if key not in f:
            return None
        json_str = f[key]['json_data'][()].decode('utf-8')  # 读取字符串并解码
        return json.loads(json_str)

# 使用示例
result = load_json_by_key('kv_store.h5', 'a1b2c3')
print(type(result))
print(result)
