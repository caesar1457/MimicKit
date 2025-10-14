"""
将 K1 运动数据转换为 MimicKit Motion 格式

输入格式（字典）:
  - fps: float
  - root_pos: (N, 3) 
  - root_rot: (N, 4) 四元数 [w, x, y, z] 或 [x, y, z, w]
  - dof_pos: (N, 22) 关节角度

输出格式（Motion对象）:
  - frames: (N, 28) [root_pos(3), root_rot_expmap(3), dof_pos(22)]
"""

import pickle
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mimickit.anim.motion import Motion
import mimickit.util.torch_util as torch_util
import torch


def quat_to_exp_map(quat):
    """
    将四元数转换为指数映射（3D旋转向量）
    
    Args:
        quat: (N, 4) numpy array, 四元数 [x, y, z, w] 或 [w, x, y, z]
    
    Returns:
        exp_map: (N, 3) numpy array, 指数映射
    """
    # 转为 torch tensor
    if isinstance(quat, np.ndarray):
        quat_torch = torch.from_numpy(quat).float()
    else:
        quat_torch = quat
    
    # 检查四元数格式 (wxyz vs xyzw)
    # 通常 w 应该在 [-1, 1] 范围且是主分量
    if quat.shape[-1] == 4:
        # 假设是 [x, y, z, w] 格式，转为 [w, x, y, z]
        if np.abs(quat[:, 3]).mean() > np.abs(quat[:, 0]).mean():
            # 最后一列是 w
            quat_torch = quat_torch[:, [3, 0, 1, 2]]
    
    # 归一化
    quat_torch = quat_torch / torch.norm(quat_torch, dim=-1, keepdim=True)
    
    # 转为指数映射
    exp_map = torch_util.quat_to_exp_map(quat_torch)
    
    return exp_map.numpy()


def convert_motion(input_file, output_file=None):
    """
    转换运动数据
    
    Args:
        input_file: 输入pkl文件路径
        output_file: 输出pkl文件路径（如果为None，会自动生成）
    """
    print(f"Loading: {input_file}")
    
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    # 提取数据
    fps = data['fps']
    root_pos = data['root_pos']  # (N, 3)
    root_rot_quat = data['root_rot']  # (N, 4)
    dof_pos = data['dof_pos']  # (N, 22)
    
    num_frames = root_pos.shape[0]
    
    print(f"\nInput data:")
    print(f"  Frames: {num_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {num_frames/fps:.2f}s")
    print(f"  root_pos shape: {root_pos.shape}")
    print(f"  root_rot shape: {root_rot_quat.shape}")
    print(f"  dof_pos shape: {dof_pos.shape}")
    
    # 转换四元数为指数映射
    print("\nConverting quaternion to exponential map...")
    root_rot_exp = quat_to_exp_map(root_rot_quat)
    print(f"  root_rot_exp shape: {root_rot_exp.shape}")
    
    # 组合成 MimicKit 格式: [root_pos(3), root_rot(3), dof_pos(22)]
    frames = np.concatenate([
        root_pos,      # (N, 3)
        root_rot_exp,  # (N, 3)
        dof_pos        # (N, 22)
    ], axis=1)  # (N, 28)
    
    print(f"\nOutput frames shape: {frames.shape}")
    
    # 创建 Motion 对象
    motion = Motion(
        frames=frames,
        fps=fps,
        loop_mode="wrap"  # 可以是 "wrap" 或 "none"
    )
    
    # 确定输出文件名
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(
            os.path.dirname(input_file),
            f"{base_name}_converted.pkl"
        )
    
    # 保存
    print(f"\nSaving to: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(motion, f)
    
    print(f"\n✅ Conversion complete!")
    print(f"   Input:  {input_file}")
    print(f"   Output: {output_file}")
    print(f"   Frames: {motion.frames.shape[0]}")
    print(f"   FPS: {motion.fps}")
    
    return output_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert K1 motion data to MimicKit format')
    parser.add_argument('--input', type=str, 
                       default='data/motions/k1/k1_K1_run_kick_mix_penalty_aligned_time_1013-1538.pkl',
                       help='Input motion file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output motion file (default: auto-generate from input name)')
    
    args = parser.parse_args()
    
    convert_motion(args.input, args.output)


if __name__ == '__main__':
    main()

