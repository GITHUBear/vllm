import h5py
import json
import numpy as np

metafile = "./meta.hdf5"
num_tp = 8
with h5py.File(metafile, 'r') as f:
    print(f"item cnt: {len(f.keys())}")
    for key in f.keys():
        print(key)
    #     item = f[key]
    #     json_str = item['meta'][()].decode('utf-8')
    #     meta = json.loads(json_str)

    #     path = meta['path']
    #     offset = meta['offset']
    #     print(f"key:{key}  -----> meta: [path:{path} offset:{offset}]")
        # tp = 0
        # actual_path = path + f"_tp{tp}.hdf5"
        # with h5py.File(actual_path, 'r') as f2:
        #     for k2 in f2.keys():
        #         print(f"\ttp-[{tp}]-{k2}:{np.array(f2[k2]).shape}")