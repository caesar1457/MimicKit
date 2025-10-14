"""
创建 K1 运动数据模板

这个脚本帮助您创建一个简单的 K1 运动数据文件模板
可以用作测试或作为创建实际运动数据的参考

使用方法:
    python tools/create_k1_motion_template.py --output data/motions/k1/k1_walk.pkl
"""

import argparse
import numpy as np
import pickle
import os
import sys

# 添加 mimickit 到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mimickit.anim.motion import Motion


def create_simple_walk_motion(duration=2.0, fps=60):
    """
    创建一个简单的步行运动示例
    
    Args:
        duration: 运动持续时间（秒）
        fps: 帧率
    
    Returns:
        Motion 对象
    """
    num_frames = int(duration * fps)
    dt = 1.0 / fps
    
    # K1 关节顺序（22个关节）
    joint_names = [
        "Head_yaw", "Head_pitch",
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
        "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", 
        "Left_Ankle_Pitch", "Left_Ankle_Roll",
        "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
        "Right_Ankle_Pitch", "Right_Ankle_Roll"
    ]
    
    num_joints = len(joint_names)
    frames = []
    
    print(f"Creating {num_frames} frames for K1 walking motion...")
    
    for i in range(num_frames):
        t = i * dt
        
        # Root position: 前进 + 轻微上下摆动
        root_pos = np.array([
            0.3 * t,  # x: 前进
            0.0,      # y: 不横向移动
            0.65 + 0.02 * np.sin(2 * np.pi * t)  # z: 轻微上下
        ])
        
        # Root rotation (exponential map): 保持直立
        root_rot = np.array([0.0, 0.0, 0.0])
        
        # 创建关节角度（简单的正弦波模拟步行）
        joint_angles = np.zeros(num_joints)
        
        # 头部：保持中立
        joint_angles[0] = 0.0  # Head_yaw
        joint_angles[1] = 0.0  # Head_pitch
        
        # 手臂：轻微摆动
        arm_swing = 0.3 * np.sin(2 * np.pi * t)
        joint_angles[2] = arm_swing    # Left_Shoulder_Pitch
        joint_angles[3] = 0.0          # Left_Shoulder_Roll
        joint_angles[4] = 0.5          # Left_Elbow_Pitch (弯曲)
        joint_angles[5] = 0.0          # Left_Elbow_Yaw
        
        joint_angles[6] = -arm_swing   # Right_Shoulder_Pitch (反向)
        joint_angles[7] = 0.0          # Right_Shoulder_Roll
        joint_angles[8] = 0.5          # Right_Elbow_Pitch
        joint_angles[9] = 0.0          # Right_Elbow_Yaw
        
        # 腿部：模拟步行
        leg_phase = 2 * np.pi * t
        
        # 左腿
        joint_angles[10] = -0.3 * np.sin(leg_phase)       # Left_Hip_Pitch
        joint_angles[11] = 0.0                            # Left_Hip_Roll
        joint_angles[12] = 0.0                            # Left_Hip_Yaw
        joint_angles[13] = 0.6 * (1 + np.sin(leg_phase)) # Left_Knee_Pitch
        joint_angles[14] = 0.2 * np.sin(leg_phase)       # Left_Ankle_Pitch
        joint_angles[15] = 0.0                            # Left_Ankle_Roll
        
        # 右腿（反相）
        joint_angles[16] = 0.3 * np.sin(leg_phase)        # Right_Hip_Pitch
        joint_angles[17] = 0.0                            # Right_Hip_Roll
        joint_angles[18] = 0.0                            # Right_Hip_Yaw
        joint_angles[19] = 0.6 * (1 - np.sin(leg_phase)) # Right_Knee_Pitch
        joint_angles[20] = -0.2 * np.sin(leg_phase)      # Right_Ankle_Pitch
        joint_angles[21] = 0.0                            # Right_Ankle_Roll
        
        # 组合成一帧: [root_pos(3), root_rot(3), joints(22)] = 28维
        frame = np.concatenate([root_pos, root_rot, joint_angles])
        frames.append(frame)
    
    frames = np.array(frames)
    print(f"Frame shape: {frames.shape}")  # 应该是 (num_frames, 28)
    
    # 创建 Motion 对象
    motion = Motion(frames=frames, fps=fps, loop_mode="wrap")
    
    return motion


def main():
    parser = argparse.ArgumentParser(description='Create K1 motion template')
    parser.add_argument('--output', type=str, default='data/motions/k1/k1_walk.pkl',
                       help='Output pickle file path')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='Motion duration in seconds')
    parser.add_argument('--fps', type=int, default=60,
                       help='Frames per second')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建运动
    motion = create_simple_walk_motion(duration=args.duration, fps=args.fps)
    
    # 保存
    with open(args.output, 'wb') as f:
        pickle.dump(motion, f)
    
    print(f"\n✅ Motion file created: {args.output}")
    print(f"   Duration: {args.duration}s")
    print(f"   FPS: {args.fps}")
    print(f"   Frames: {motion.get_num_frames()}")
    print(f"\n📺 View the motion with:")
    print(f"   python mimickit/run.py --arg_file args/view_motion_k1_args.txt --visualize true")


if __name__ == '__main__':
    main()


