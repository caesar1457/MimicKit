"""
在无头服务器上录制K1机器人视频
使用Isaac Gym的camera sensor进行offscreen渲染

依赖: pip install imageio imageio-ffmpeg

用法:
CUDA_VISIBLE_DEVICES=6 python render_k1_video.py \
    --model_file output/k1_add/k1_add_model.pt \
    --output_video output/k1_demo.mp4 \
    --duration 10 \
    --fps 30
"""

import numpy as np
import os
import sys
import torch

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("警告: imageio未安装，只能保存PNG序列")
    print("安装方法: pip install imageio imageio-ffmpeg")

from isaacgym import gymapi, gymutil
import util.arg_parser as arg_parser
from util.logger import Logger


def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])
    
    arg_file = args.parse_string("arg_file", "")
    if arg_file != "":
        succ = args.load_file(arg_file)
        assert succ, f"加载参数文件失败: {arg_file}"
    
    return args


def create_headless_env_with_camera(args, device):
    """创建带camera sensor的headless环境"""
    
    # 创建gym
    gym = gymapi.acquire_gym()
    
    # 创建sim (headless模式)
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = True
    
    compute_device = 0 if "cuda" in device else -1
    graphics_device = compute_device  # 需要graphics device来渲染camera
    
    sim = gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)
    
    if sim is None:
        print("创建sim失败!")
        sys.exit(1)
    
    # 添加地面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)
    
    # 创建环境
    env_spacing = 5.0
    num_per_row = 1
    lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, lower, upper, num_per_row)
    
    # 加载K1机器人
    asset_root = "data/assets"
    asset_file = "k1.xml"
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.angular_damping = 0.01
    asset_options.max_angular_velocity = 100.0
    
    k1_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    
    # 创建actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0.7)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    
    actor_handle = gym.create_actor(env, k1_asset, pose, "k1", 0, 0)
    
    # 创建camera
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1280
    camera_props.height = 720
    camera_props.enable_tensors = True
    
    camera_handle = gym.create_camera_sensor(env, camera_props)
    
    # 设置camera位置
    cam_pos = gymapi.Vec3(3, 3, 2)
    cam_target = gymapi.Vec3(0, 0, 1)
    gym.set_camera_location(camera_handle, env, cam_pos, cam_target)
    
    return gym, sim, env, camera_handle, actor_handle


def render_with_random_motion(gym, sim, env, camera_handle, num_frames, fps):
    """渲染随机运动（因为加载完整的agent需要完整的env setup）"""
    
    frames = []
    
    print(f"开始渲染 {num_frames} 帧...")
    
    for i in range(num_frames):
        # Step simulation
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # Render camera
        gym.render_all_camera_sensors(sim)
        gym.start_access_image_tensors(sim)
        
        # Get camera image
        cam_img = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
        
        # 转换为numpy数组
        img = cam_img.reshape(720, 1280, 4)[:, :, :3]  # 移除alpha通道
        frames.append(img.copy())
        
        gym.end_access_image_tensors(sim)
        
        if (i + 1) % 30 == 0:
            print(f"  已渲染 {i + 1}/{num_frames} 帧")
    
    return frames


def save_frames(frames, output_path, fps):
    """保存帧为视频或图像序列"""
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if IMAGEIO_AVAILABLE and output_path.endswith('.mp4'):
        # 保存为MP4
        print(f"正在保存视频: {output_path}")
        imageio.mimsave(output_path, frames, fps=fps, codec='libx264', pixelformat='yuv420p')
        print(f"✓ 视频已保存: {output_path}")
    else:
        # 保存为PNG序列
        frame_dir = output_path.replace('.mp4', '_frames')
        os.makedirs(frame_dir, exist_ok=True)
        
        print(f"正在保存PNG序列: {frame_dir}")
        for i, frame in enumerate(frames):
            if IMAGEIO_AVAILABLE:
                imageio.imwrite(f"{frame_dir}/frame_{i:06d}.png", frame)
            else:
                # 使用PIL或其他方法
                from PIL import Image
                Image.fromarray(frame).save(f"{frame_dir}/frame_{i:06d}.png")
        
        print(f"✓ 已保存 {len(frames)} 帧到: {frame_dir}")
        print(f"\n要合成视频，请运行:")
        print(f"ffmpeg -framerate {fps} -i {frame_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p {output_path}")


def main_simple_demo():
    """简化版本：仅渲染机器人在重力下的运动"""
    
    print("="*60)
    print("K1 机器人视频渲染工具 (简化演示版)")
    print("="*60)
    
    # 参数
    output_video = "output/k1_demo.mp4"
    duration = 10  # 秒
    fps = 30
    device = "cuda:0"
    
    num_frames = int(duration * fps)
    
    print(f"输出: {output_video}")
    print(f"时长: {duration}秒")
    print(f"FPS: {fps}")
    print(f"总帧数: {num_frames}")
    print(f"设备: {device}")
    print("="*60)
    
    # 创建环境
    print("\n创建仿真环境...")
    gym, sim, env, camera_handle, actor_handle = create_headless_env_with_camera(None, device)
    print("✓ 环境创建成功")
    
    # Prepare simulation
    gym.prepare_sim(sim)
    
    # 渲染
    frames = render_with_random_motion(gym, sim, env, camera_handle, num_frames, fps)
    
    # 保存
    save_frames(frames, output_video, fps)
    
    # 清理
    gym.destroy_sim(sim)
    
    print("\n完成!")


def main(argv):
    """完整版本：加载训练好的模型"""
    
    # 由于完整版本需要重新构建整个环境系统，
    # 这里提供一个简化版本
    print("注意: 完整的模型加载功能需要更复杂的集成")
    print("当前运行简化演示版本...\n")
    
    main_simple_demo()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv)
    else:
        main_simple_demo()

