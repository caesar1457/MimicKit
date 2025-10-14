"""
简单版本：录制 K1 运动视频（kinematic 模式）

使用方法:
    python tools/record_k1_motion_simple.py
"""

import argparse
import os
import sys
import numpy as np
import pickle
import imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import isaacgym.gymapi as gymapi
from mimickit.anim.motion import Motion
import mimickit.util.torch_util as torch_util
import torch


def record_motion_video_simple(motion_file, output_video, duration=10.0, fps=30, 
                               width=1920, height=1080):
    """
    录制运动视频（简化版本）
    """
    
    print(f"\n{'='*60}")
    print(f"Recording K1 Motion Video")
    print(f"{'='*60}")
    
    # 1. 加载运动数据
    print(f"\n1. Loading motion: {motion_file}")
    with open(motion_file, 'rb') as f:
        motion = pickle.load(f)
    
    num_frames = motion.frames.shape[0]
    motion_fps = motion.fps
    motion_duration = num_frames / motion_fps
    
    print(f"   Frames: {num_frames}")
    print(f"   FPS: {motion_fps}")
    print(f"   Duration: {motion_duration:.2f}s")
    
    # 2. 初始化 IsaacGym
    print(f"\n2. Initializing IsaacGym...")
    gym = gymapi.acquire_gym()
    
    # 创建仿真参数
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
    sim_params.use_gpu_pipeline = False  # 使用 CPU pipeline 以便使用简单 API
    
    # 创建仿真（headless）
    sim = gym.create_sim(0, 0, gymapi.SimType.SIM_PHYSX, sim_params)
    
    # 添加地面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)
    
    # 3. 加载 K1 资产
    print(f"   Loading K1 asset...")
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.disable_gravity = True  # kinematic 模式
    asset = gym.load_asset(sim, "data/assets/k1", "K1_serial.xml", asset_options)
    
    # 4. 创建环境
    print(f"   Creating environment...")
    env = gym.create_env(sim, gymapi.Vec3(-5, -5, 0), gymapi.Vec3(5, 5, 5), 1)
    
    # 创建 actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0.65)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    actor_handle = gym.create_actor(env, asset, pose, "k1", 0, 0)
    
    # 设置为 kinematic
    props = gym.get_actor_dof_properties(env, actor_handle)
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(10000.0)
    props["damping"].fill(1000.0)
    gym.set_actor_dof_properties(env, actor_handle, props)
    
    gym.prepare_sim(sim)
    
    # 5. 设置相机
    print(f"   Setting up camera ({width}x{height})...")
    cam_props = gymapi.CameraProperties()
    cam_props.width = width
    cam_props.height = height
    camera = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(camera, env, 
                           gymapi.Vec3(3, 3, 2),
                           gymapi.Vec3(0, 0, 0.5))
    
    # 6. 录制视频
    num_video_frames = int(duration * fps)
    print(f"\n3. Recording {num_video_frames} frames...")
    
    frames_list = []
    
    for i in range(num_video_frames):
        # 计算当前运动帧索引
        motion_time = (i / fps) % motion_duration
        frame_idx = int(motion_time * motion_fps) % num_frames
        
        # 获取运动帧数据
        frame_data = motion.frames[frame_idx]
        root_pos = frame_data[0:3]
        root_rot_exp = frame_data[3:6]
        dof_pos = frame_data[6:28]
        
        # 转换旋转
        root_rot_quat = torch_util.exp_map_to_quat(
            torch.tensor(root_rot_exp, dtype=torch.float32)
        ).numpy()
        
        # 设置 DOF 位置
        gym.set_actor_dof_position_targets(env, actor_handle, dof_pos.tolist())
        
        # 设置根位置（通过设置刚体状态）
        # 注意：free joint 通常是第一个刚体
        root_state = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)
        root_state['pose']['p'].fill((root_pos[0], root_pos[1], root_pos[2]))
        root_state['pose']['r'].fill((root_rot_quat[1], root_rot_quat[2], 
                                      root_rot_quat[3], root_rot_quat[0]))
        gym.set_actor_rigid_body_states(env, actor_handle, root_state, gymapi.STATE_POS)
        
        # 仿真一步
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # 渲染
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        
        # 获取图像
        img = gym.get_camera_image(sim, env, camera, gymapi.IMAGE_COLOR)
        img_rgb = img.reshape(height, width, 4)[:, :, :3]
        frames_list.append(img_rgb)
        
        if (i + 1) % 30 == 0:
            print(f"   Progress: {i+1}/{num_video_frames}")
    
    # 7. 保存视频
    print(f"\n4. Saving video...")
    os.makedirs(os.path.dirname(output_video) or '.', exist_ok=True)
    imageio.mimsave(output_video, frames_list, fps=fps, quality=8)
    
    # 清理
    gym.destroy_sim(sim)
    
    print(f"\n{'='*60}")
    print(f"✅ Video saved: {output_video}")
    print(f"   Duration: {duration}s | FPS: {fps} | Frames: {num_video_frames}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion_file', default='data/motions/k1/k1_walk.pkl')
    parser.add_argument('--output_video', default='output/k1_motion.mp4')
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    
    args = parser.parse_args()
    
    record_motion_video_simple(
        args.motion_file, args.output_video, 
        args.duration, args.fps,
        args.width, args.height
    )


if __name__ == '__main__':
    main()

