"""
修复 numpy 2.0 pkl 文件到 numpy 1.x 兼容格式

使用方法:
    python tools/fix_numpy_compat.py \
        --input data/motions/k1/k1_K1_run_kick_mix_penalty_aligned_time_1013-1538.pkl \
        --output data/motions/k1/k1_original_fixed.pkl
"""

import sys
import pickle
import argparse

# 兼容性补丁
class NumpyCompatUnpickler(pickle.Unpickler):
    """处理 numpy 2.0 到 1.x 的兼容性"""
    
    def find_class(self, module, name):
        # 将 numpy._core 重定向到 numpy.core
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def load_with_compat(filename):
    """兼容性加载 pickle 文件"""
    with open(filename, 'rb') as f:
        return NumpyCompatUnpickler(f).load()


def fix_pkl_file(input_file, output_file):
    """
    修复 pkl 文件的 numpy 兼容性
    """
    print(f"Loading: {input_file}")
    data = load_with_compat(input_file)
    
    print(f"Data type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for k, v in data.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}")
    
    print(f"\nSaving to: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    print(f"✅ Fixed file saved!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                       default='data/motions/k1/k1_K1_run_kick_mix_penalty_aligned_time_1013-1538.pkl')
    parser.add_argument('--output', type=str,
                       default='data/motions/k1/k1_original_fixed.pkl')
    
    args = parser.parse_args()
    fix_pkl_file(args.input, args.output)


if __name__ == '__main__':
    main()


