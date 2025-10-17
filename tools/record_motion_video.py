"""
录制机器人运动数据为视频（无需可视化窗口）

使用 IsaacGym 的离屏渲染功能录制视频
支持 K1 和 G1 两种机器人

使用方法:
    # 使用默认设置（自动生成输出文件名）
    python tools/record_motion_video.py \
        --motion_file data/motions/g1/g1_walk.pkl \
        --robot_type g1 \
        --duration 10
    
    # 指定输出路径
    python tools/record_motion_video.py \
        --motion_file data/motions/k1/k1_walk.pkl \
        --robot_type k1 \
        --output_video video/my_video.mp4 \
        --duration 10
        
输出视频自动命名格式：video/时间戳_机器人类型_运动名称.mp4
例如：video/20251015_143022_g1_g1_walk.mp4
"""

import argparse
import os
import sys
import numpy as np
import pickle
import imageio
from datetime import datetime

# 添加 mimickit 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import isaacgym.gymapi as gymapi
import isaacgym.gymutil as gymutil
from isaacgym import gymtorch
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
    sim_params.use_gpu_pipeline = False
    
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


def load_robot_asset(gym, sim, robot_type="k1"):
    """加载机器人模型
    
    Args:
        gym: IsaacGym 实例
        sim: 仿真环境
        robot_type: 机器人类型，"k1" 或 "g1"
    """
    # 根据机器人类型选择资产
    robot_configs = {
        "k1": {
            "asset_root": "data/assets/k1",
            "asset_file": "K1_serial.xml"
        },
        "g1": {
            "asset_root": "data/assets/g1",
            "asset_file": "g1.xml"
        }
    }
    
    if robot_type not in robot_configs:
        raise ValueError(f"Unsupported robot type: {robot_type}. Supported types: {list(robot_configs.keys())}")
    
    config = robot_configs[robot_type]
    asset_root = config["asset_root"]
    asset_file = config["asset_file"]
    
    asset_options = gymapi.AssetOptions()
    asset_options.angular_damping = 0.01
    asset_options.max_angular_velocity = 100.0
    asset_options.fix_base_link = False
    
    print(f"Loading {robot_type.upper()} asset: {asset_root}/{asset_file}")
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    
    return asset


def create_env_with_actor(gym, sim, asset, robot_type="k1", env_id=0):
    """创建环境并添加机器人角色
    
    Args:
        gym: IsaacGym 实例
        sim: 仿真环境
        asset: 机器人资产
        robot_type: 机器人类型
        env_id: 环境ID
    """
    # 创建环境
    env_spacing = 5.0
    num_per_row = 1
    lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, lower, upper, num_per_row)
    
    # 根据机器人类型设置初始高度
    initial_heights = {
        "k1": 0.65,
        "g1": 0.8
    }
    initial_height = initial_heights.get(robot_type, 0.65)
    
    # 创建角色
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, initial_height)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    actor = gym.create_actor(env, asset, pose, robot_type, env_id, 0, 0)
    
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
    
    # 初始相机位置（之后会更新跟随机器人）
    cam_pos = gymapi.Vec3(3.0, 3.0, 2.0)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
    gym.set_camera_location(camera_handle, env, cam_pos, cam_target)
    
    return camera_handle


def update_camera_follow_robot(gym, env, camera_handle, robot_pos, camera_offset=(3.0, 3.0, 2.0)):
    """
    更新相机位置以跟随机器人
    
    Args:
        gym: IsaacGym 实例
        env: 环境
        camera_handle: 相机句柄
        robot_pos: 机器人根位置 [x, y, z]
        camera_offset: 相机相对机器人的偏移 (dx, dy, dz)
    """
    # 相机位置跟随机器人，保持固定偏移
    cam_pos = gymapi.Vec3(
        robot_pos[0] + camera_offset[0],
        robot_pos[1] + camera_offset[1],
        robot_pos[2] + camera_offset[2]
    )
    
    # 相机目标点为机器人位置（稍微高一点看机器人中心）
    cam_target = gymapi.Vec3(
        robot_pos[0],
        robot_pos[1],
        robot_pos[2] + 0.5
    )
    
    gym.set_camera_location(camera_handle, env, cam_pos, cam_target)


def set_actor_state_from_frame(gym, sim, env, actor, frame, root_state_tensor, dof_state_tensor):
    """
    从运动帧设置角色状态（使用 Tensor API）
    
    Args:
        gym: IsaacGym 实例
        sim: 仿真环境
        env: 环境句柄
        actor: 角色句柄
        frame: 运动帧数据 [root_pos(3), root_rot_expmap(3), dof_pos(N)]
        root_state_tensor: root 状态 tensor，形状 [num_actors, 13]
        dof_state_tensor: DOF 状态 tensor，形状 [num_dofs, 2]
    """
    # 提取数据
    root_pos = frame[0:3]
    root_rot_exp = frame[3:6]
    dof_pos = frame[6:]
    
    # 转换旋转（指数映射 -> 四元数）
    root_rot_exp_torch = torch.tensor(root_rot_exp, dtype=torch.float32)
    root_rot_quat = torch_util.exp_map_to_quat(root_rot_exp_torch)
    # 四元数格式：[w, x, y, z] -> [x, y, z, w]
    quat_xyzw = torch.tensor([root_rot_quat[1], root_rot_quat[2], 
                               root_rot_quat[3], root_rot_quat[0]], dtype=torch.float32)
    
    # 设置 root 状态 [pos(3), rot(4), vel(3), ang_vel(3)] = 13
    # root_state_tensor 形状是 [num_actors, 13]，我们只有一个 actor，索引为 0
    root_state_tensor[0, 0:3] = torch.tensor(root_pos, dtype=torch.float32)
    root_state_tensor[0, 3:7] = quat_xyzw
    root_state_tensor[0, 7:10] = 0.0  # 线速度
    root_state_tensor[0, 10:13] = 0.0  # 角速度
    
    # 设置 DOF 状态 [pos, vel] for each DOF
    num_dofs = gym.get_actor_dof_count(env, actor)
    for i in range(min(num_dofs, len(dof_pos))):
        dof_state_tensor[i, 0] = float(dof_pos[i])  # 位置
        dof_state_tensor[i, 1] = 0.0  # 速度


def record_motion_video(motion_file, output_video, robot_type="k1", duration=10.0, fps=30, 
                       width=1920, height=1080, camera_follow=True):
    """
    录制运动视频
    
    Args:
        motion_file: 运动数据 pkl 文件
        output_video: 输出视频路径
        robot_type: 机器人类型，"k1" 或 "g1"
        duration: 录制时长（秒）
        fps: 视频帧率
        width, height: 视频分辨率
        camera_follow: 是否启用相机跟随机器人
    """
    
    print(f"\n{'='*60}")
    print(f"Recording Motion Video")
    print(f"{'='*60}")
    
    # 加载运动数据前，先注册模块别名（解决 pickle 反序列化问题）
    # pickle 文件中的类路径是 anim.motion.Motion，需要映射到 mimickit.anim
    import mimickit.anim as anim_module
    sys.modules['anim'] = anim_module
    sys.modules['anim.motion'] = anim_module.motion
    
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
    print(f"   Loading {robot_type.upper()} asset...")
    asset = load_robot_asset(gym, sim, robot_type)
    
    # 创建环境和角色
    print(f"   Creating environment...")
    env, actor = create_env_with_actor(gym, sim, asset, robot_type, 0)
    
    # 准备仿真
    gym.prepare_sim(sim)
    
    # 获取状态 tensors（用于高效更新机器人状态）
    print(f"   Acquiring state tensors...")
    root_state_tensor_raw = gym.acquire_actor_root_state_tensor(sim)
    dof_state_tensor_raw = gym.acquire_dof_state_tensor(sim)
    
    root_state_tensor = gymtorch.wrap_tensor(root_state_tensor_raw)
    dof_state_tensor = gymtorch.wrap_tensor(dof_state_tensor_raw)
    
    # 获取 actor 索引
    actor_indices = torch.tensor([0], dtype=torch.int32)
    
    # 设置相机
    print(f"   Setting up camera ({width}x{height})...")
    camera = setup_camera(gym, env, width, height)
    
    # 计算需要录制的帧数
    num_video_frames = int(duration * fps)
    print(f"\n3. Recording {num_video_frames} frames at {fps} FPS...")
    print(f"   Camera follow: {'Enabled' if camera_follow else 'Disabled (fixed camera)'}")
    
    frames = []
    
    for i in range(num_video_frames):
        # 计算当前运动时间（循环播放）
        motion_time = (i / fps) % motion_duration
        motion_frame_idx = int(motion_time * motion_fps) % num_frames
        
        # 获取运动帧
        frame = motion.frames[motion_frame_idx]
        
        # 设置角色状态
        set_actor_state_from_frame(gym, sim, env, actor, frame, root_state_tensor, dof_state_tensor)
        
        # 应用状态更新到仿真
        gym.set_actor_root_state_tensor_indexed(sim, 
                                                  gymtorch.unwrap_tensor(root_state_tensor),
                                                  gymtorch.unwrap_tensor(actor_indices), len(actor_indices))
        gym.set_dof_state_tensor_indexed(sim,
                                         gymtorch.unwrap_tensor(dof_state_tensor),
                                         gymtorch.unwrap_tensor(actor_indices), len(actor_indices))
        
        # 更新相机位置跟随机器人（如果启用）
        if camera_follow:
            robot_pos = frame[0:3]  # [x, y, z]
            update_camera_follow_robot(gym, env, camera, robot_pos)
        
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
    parser = argparse.ArgumentParser(description='Record robot motion as video')
    parser.add_argument('--motion_file', type=str,
                       default='data/motions/g1/g1_walk.pkl',
                       help='Input motion file')
    parser.add_argument('--robot_type', type=str,
                       default='g1',
                       choices=['k1', 'g1'],
                       help='Robot type: k1 or g1')
    parser.add_argument('--output_video', type=str,
                       default=None,
                       help='Output video file (default: auto-generated in video/ folder)')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Video duration in seconds')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video FPS')
    parser.add_argument('--width', type=int, default=1920,
                       help='Video width')
    parser.add_argument('--height', type=int, default=1080,
                       help='Video height')
    parser.add_argument('--no_camera_follow', action='store_true',
                       help='Disable camera following robot (use fixed camera)')
    
    args = parser.parse_args()
    
    # 如果没有指定输出路径，自动生成：video/时间_机器人_motion名.mp4
    if args.output_video is None:
        # 提取 motion 文件名（不含扩展名）
        motion_basename = os.path.basename(args.motion_file)
        motion_name = os.path.splitext(motion_basename)[0]
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成输出文件名：时间_机器人_motion名.mp4
        output_filename = f"{timestamp}_{args.robot_type}_{motion_name}.mp4"
        args.output_video = os.path.join("video", output_filename)
        
        print(f"Auto-generated output path: {args.output_video}")
    
    record_motion_video(
        motion_file=args.motion_file,
        output_video=args.output_video,
        robot_type=args.robot_type,
        duration=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        camera_follow=not args.no_camera_follow
    )


if __name__ == '__main__':
    main()

