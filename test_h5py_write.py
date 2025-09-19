import h5py
import json
import numpy as np

def save_kv_to_hdf5(kv_dict, filepath):
    """
    将 {hash: json_dict} 字典保存到 HDF5 文件中
    """
    with h5py.File(filepath, 'w') as f:
        for key, value in kv_dict.items():
            # 创建以 hash 为名的 group
            grp = f.create_group(key)
            # 将 JSON 序列化为字符串并存为 dataset
            json_str = json.dumps(value, ensure_ascii=False)  # 支持中文
            grp.create_dataset('json_data', data=np.bytes_(json_str))

# 示例数据
data = {
    "a1b2c3": [{"tokens":[1,2,3], "path":"a", "offset":1}, {"tokens":[4,5,6], "path":"b", "offset":12}],
    "x9y8z7": [{"tokens":[7], "path":"c", "offset":1}],
}

save_kv_to_hdf5(data, 'kv_store.h5')
