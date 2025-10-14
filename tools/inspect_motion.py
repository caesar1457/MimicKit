import pickle
import numpy as np
import sys

motion_file = sys.argv[1] if len(sys.argv) > 1 else 'data/motions/k1/k1_K1_run_kick_mix_penalty_aligned_time_1013-1538.pkl'

print(f"Loading: {motion_file}")
with open(motion_file, 'rb') as f:
    data = pickle.load(f)

print(f"\nType: {type(data)}")

if isinstance(data, dict):
    print(f"\nDictionary Keys: {list(data.keys())}")
    print("\nDetails:")
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        elif hasattr(v, '__len__'):
            print(f"  {k}: len={len(v)}, type={type(v)}")
        else:
            print(f"  {k}: {v} (type={type(v)})")
else:
    print(f"\nObject attributes: {dir(data)}")
    if hasattr(data, '_frames'):
        print(f"Frames shape: {data._frames.shape}")
    if hasattr(data, '_fps'):
        print(f"FPS: {data._fps}")
    if hasattr(data, 'get_num_frames'):
        print(f"Num frames: {data.get_num_frames()}")


