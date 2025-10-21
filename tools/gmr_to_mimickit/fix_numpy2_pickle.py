#!/usr/bin/env python3
"""
修复用 numpy 2.0+ 保存的 pickle 文件，使其能在 numpy 1.x 下读取
通过重新映射模块名称来解决兼容性问题
"""
import pickle
import sys
import io

class NumpyUnpickler(pickle.Unpickler):
    """自定义 Unpickler 来处理 numpy 2.0 的模块重命名"""
    
    def find_class(self, module, name):
        # 将 numpy._core 重新映射到 numpy.core
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        elif module.startswith('numpy._'):
            # 处理其他 numpy._ 开头的私有模块
            module = module.replace('numpy._', 'numpy.')
        
        return super().find_class(module, name)

def load_numpy2_pickle(file_path):
    """使用自定义 unpickler 加载 numpy 2.0+ 保存的 pickle 文件"""
    with open(file_path, 'rb') as f:
        unpickler = NumpyUnpickler(f)
        return unpickler.load()

def resave_pickle(input_path, output_path):
    """重新保存 pickle 文件，使用兼容的协议"""
    print(f"正在加载文件: {input_path}")
    
    try:
        data = load_numpy2_pickle(input_path)
        print("✅ 文件加载成功！")
        
        # 显示数据信息
        if isinstance(data, dict):
            print(f"\n数据类型: 字典")
            print(f"键: {list(data.keys())}")
            if 'root_pos' in data:
                print(f"root_pos shape: {data['root_pos'].shape}")
            if 'fps' in data:
                print(f"fps: {data['fps']}")
        
        # 用较低的协议重新保存
        print(f"\n正在保存到: {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(data, f, protocol=4)
        
        print("✅ 文件已成功转换！")
        print("\n现在可以在 numpy 1.x 环境下使用该文件了。")
        
    except Exception as e:
        print(f"❌ 错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python fix_numpy2_pickle.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = resave_pickle(input_file, output_file)
    sys.exit(0 if success else 1)


