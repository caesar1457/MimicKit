"""
录制 K1 运动数据为视频（无需可视化窗口）

使用 IsaacGym 的离屏渲染功能录制视频

使用方法:
    python tools/record_motion_video.py \
        --motion_file data/motions/k1/k1_walk.pkl \
        --output_video output/k1_motion.mp4 \
        --duration 10
"""

import argparse
import os
import sys
import numpy as np
import pickle
import imageio

# 添加 mimickit 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import isaacgym.gymapi as gymapi
import isaacgym.gymutil as gymutil
from mimickit.anim.motion import Motion
import mimickit.util.torch_util as torch_util
import torch


def create_sim(gym, num_envs=1):
    """创建仿真环境（离屏渲染）"""
    
    # 仿真参数
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0  # 60 Hz
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    # PhysX 参数
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = True
    
    # 创建仿真（headless，用于离屏渲染）
    compute_device = 0
    graphics_device = 0
    sim = gym.create_sim(compute_device, graphics_device, 
                        gymapi.SimType.SIM_PHYSX, sim_params)
    
    if sim is None:
        raise Exception("Failed to create sim")
    
    # 添加地面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution = 0.0
    gym.add_ground(sim, plane_params)
    
    return sim


def load_k1_asset(gym, sim):
    """加载 K1 机器人模型"""
    
    asset_root = "data/assets/k1"
    asset_file = "K1_serial.xml"
    
    asset_options = gymapi.AssetOptions()
    asset_options.angular_damping = 0.01
    asset_options.max_angular_velocity = 100.0
    asset_options.fix_base_link = False
    
    print(f"Loading asset: {asset_root}/{asset_file}")
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    
    return asset


def create_env_with_actor(gym, sim, asset, env_id=0):
    """创建环境并添加 K1 角色"""
    
    # 创建环境
    env_spacing = 5.0
    num_per_row = 1
    lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, lower, upper, num_per_row)
    
    # 创建角色
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.65)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    actor = gym.create_actor(env, asset, pose, "k1", env_id, 0, 0)
    
    # 设置颜色（可选）
    num_bodies = gym.get_actor_rigid_body_count(env, actor)
    for i in range(num_bodies):
        gym.set_rigid_body_color(env, actor, i, gymapi.MESH_VISUAL,
                                gymapi.Vec3(0.8, 0.8, 0.8))
    
    return env, actor


def setup_camera(gym, env, width=1920, height=1080):
    """设置相机用于录制"""
    
    cam_props = gymapi.CameraProperties()
    cam_props.width = width
    cam_props.height = height
    cam_props.enable_tensors = True
    
    camera_handle = gym.create_camera_sensor(env, cam_props)
    
    # 设置相机位置和朝向
    cam_pos = gymapi.Vec3(3.0, 3.0, 2.0)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
    gym.set_camera_location(camera_handle, env, cam_pos, cam_target)
    
    return camera_handle


def set_actor_state_from_frame(gym, env, actor, frame):
    """
    从运动帧设置角色状态
    
    frame: [root_pos(3), root_rot_expmap(3), dof_pos(22)]
    """
    # 提取数据
    root_pos = frame[0:3]
    root_rot_exp = frame[3:6]
    dof_pos = frame[6:28]
    
    # 转换旋转（指数映射 -> 四元数）
    root_rot_exp_torch = torch.tensor(root_rot_exp, dtype=torch.float32)
    root_rot_quat = torch_util.exp_map_to_quat(root_rot_exp_torch).numpy()
    # 四元数格式：[w, x, y, z] -> [x, y, z, w]
    quat_xyzw = [root_rot_quat[1], root_rot_quat[2], root_rot_quat[3], root_rot_quat[0]]
    
    # 设置根状态（position + rotation）
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(float(root_pos[0]), float(root_pos[1]), float(root_pos[2]))
    pose.r = gymapi.Quat(float(quat_xyzw[0]), float(quat_xyzw[1]), 
                         float(quat_xyzw[2]), float(quat_xyzw[3]))
    
    gym.set_rigid_transform(env, actor, pose)
    
    # 设置关节状态
    num_dofs = gym.get_actor_dof_count(env, actor)
    for i in range(min(num_dofs, len(dof_pos))):
        gym.set_dof_target_position(env, i, float(dof_pos[i]))


def record_motion_video(motion_file, output_video, duration=10.0, fps=30, 
                       width=1920, height=1080):
    """
    录制运动视频
    
    Args:
        motion_file: 运动数据 pkl 文件
        output_video: 输出视频路径
        duration: 录制时长（秒）
        fps: 视频帧率
        width, height: 视频分辨率
    """
    
    print(f"\n{'='*60}")
    print(f"Recording Motion Video")
    print(f"{'='*60}")
    
    # 加载运动数据
    print(f"\n1. Loading motion: {motion_file}")
    with open(motion_file, 'rb') as f:
        motion = pickle.load(f)
    
    if not isinstance(motion, Motion):
        raise ValueError("Motion file must contain a Motion object")
    
    num_frames = motion.frames.shape[0]
    motion_fps = motion.fps
    motion_duration = num_frames / motion_fps
    
    print(f"   Motion frames: {num_frames}")
    print(f"   Motion FPS: {motion_fps}")
    print(f"   Motion duration: {motion_duration:.2f}s")
    
    # 初始化 IsaacGym
    print(f"\n2. Initializing IsaacGym...")
    gym = gymapi.acquire_gym()
    
    # 创建仿真
    print(f"   Creating simulation...")
    sim = create_sim(gym, num_envs=1)
    
    # 加载资产
    print(f"   Loading K1 asset...")
    asset = load_k1_asset(gym, sim)
    
    # 创建环境和角色
    print(f"   Creating environment...")
    env, actor = create_env_with_actor(gym, sim, asset, 0)
    
    # 准备仿真
    gym.prepare_sim(sim)
    
    # 设置相机
    print(f"   Setting up camera ({width}x{height})...")
    camera = setup_camera(gym, env, width, height)
    
    # 计算需要录制的帧数
    num_video_frames = int(duration * fps)
    print(f"\n3. Recording {num_video_frames} frames at {fps} FPS...")
    
    frames = []
    
    for i in range(num_video_frames):
        # 计算当前运动时间（循环播放）
        motion_time = (i / fps) % motion_duration
        motion_frame_idx = int(motion_time * motion_fps) % num_frames
        
        # 获取运动帧
        frame = motion.frames[motion_frame_idx]
        
        # 设置角色状态
        set_actor_state_from_frame(gym, env, actor, frame)
        
        # 仿真一步
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # 渲染相机
        gym.render_all_camera_sensors(sim)
        gym.start_access_image_tensors(sim)
        
        # 获取图像
        img = gym.get_camera_image(sim, env, camera, gymapi.IMAGE_COLOR)
        
        gym.end_access_image_tensors(sim)
        
        # 转换为 RGB
        img_rgb = img.reshape(height, width, 4)[:, :, :3]
        frames.append(img_rgb)
        
        if (i + 1) % 30 == 0:
            print(f"   Progress: {i+1}/{num_video_frames} frames")
    
    # 保存视频
    print(f"\n4. Saving video to: {output_video}")
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    imageio.mimsave(output_video, frames, fps=fps, quality=8, macro_block_size=1)
    
    # 清理
    gym.destroy_sim(sim)
    
    print(f"\n{'='*60}")
    print(f"✅ Video saved successfully!")
    print(f"   File: {output_video}")
    print(f"   Duration: {duration}s")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Frames: {num_video_frames}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Record K1 motion as video')
    parser.add_argument('--motion_file', type=str,
                       default='data/motions/k1/k1_walk.pkl',
                       help='Input motion file')
    parser.add_argument('--output_video', type=str,
                       default='output/k1_motion.mp4',
                       help='Output video file')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Video duration in seconds')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video FPS')
    parser.add_argument('--width', type=int, default=1920,
                       help='Video width')
    parser.add_argument('--height', type=int, default=1080,
                       help='Video height')
    
    args = parser.parse_args()
    
    record_motion_video(
        motion_file=args.motion_file,
        output_video=args.output_video,
        duration=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height
    )


if __name__ == '__main__':
    main()

