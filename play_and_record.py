"""
Play a trained model and record video (for headless server)

用法:
python play_and_record.py \
    --model_file output/k1_add/k1_add_model.pt \
    --arg_file args/add_k1_args.txt \
    --output_video output/k1_demo.mp4 \
    --num_frames 300 \
    --num_envs 1
"""

import numpy as np
import os
import sys
import torch
import imageio

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.arg_parser as arg_parser
from util.logger import Logger

# Isaac Gym imports
from isaacgym import gymapi, gymutil


def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])

    arg_file = args.parse_string("arg_file", "")
    if arg_file != "":
        succ = args.load_file(arg_file)
        assert succ, Logger.print("Failed to load args from: " + arg_file)

    return args


def build_env(args, num_envs, device):
    env_file = args.parse_string("env_config")
    # 强制使用可视化，但不显示窗口
    env = env_builder.build_env(env_file, num_envs, device, visualize=True)
    return env


def build_agent(agent_file, env, device):
    agent = agent_builder.build_agent(agent_file, env, device)
    return agent


def record_video(agent, env, output_path, num_frames=300, camera_settings=None):
    """
    Record video of the agent playing
    
    Args:
        agent: Trained agent
        env: Environment
        output_path: Path to save video (e.g., 'output/video.mp4')
        num_frames: Number of frames to record
        camera_settings: Dict with 'pos' and 'look_at' keys (optional)
    """
    print(f"开始录制视频，共 {num_frames} 帧...")
    print(f"输出路径: {output_path}")
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 获取gym实例
    gym = env._engine._gym
    sim = env._engine._sim
    viewer = env._engine._viewer
    
    # 设置相机（如果提供了设置）
    if camera_settings is not None:
        pos = camera_settings.get('pos', [5.0, 5.0, 2.0])
        look_at = camera_settings.get('look_at', [0.0, 0.0, 1.0])
        gym.viewer_camera_look_at(viewer, None, 
                                  gymapi.Vec3(*pos),
                                  gymapi.Vec3(*look_at))
    
    # 重置环境
    obs, info = env.reset()
    
    frames = []
    
    for frame_idx in range(num_frames):
        # 获取动作
        with torch.no_grad():
            action_info = agent.decide_action(obs, info)
            action = action_info["action"]
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        
        # 渲染（更新可视化）
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        
        # 获取图像
        gym.write_viewer_image_to_file(viewer, f"/tmp/frame_{frame_idx:06d}.png")
        
        # 读取图像
        img = imageio.imread(f"/tmp/frame_{frame_idx:06d}.png")
        frames.append(img)
        
        # 删除临时文件
        os.remove(f"/tmp/frame_{frame_idx:06d}.png")
        
        if (frame_idx + 1) % 50 == 0:
            print(f"已录制 {frame_idx + 1}/{num_frames} 帧")
    
    # 保存视频
    print(f"正在保存视频到 {output_path}...")
    imageio.mimsave(output_path, frames, fps=30)
    print(f"视频录制完成！")
    
    return frames


def main(argv):
    args = load_args(argv)
    
    # 解析参数
    device = args.parse_string("device", "cuda:0")
    num_envs = args.parse_int("num_envs", 1)
    model_file = args.parse_string("model_file", "")
    output_video = args.parse_string("output_video", "output/video.mp4")
    num_frames = args.parse_int("num_frames", 300)
    
    # 相机设置
    camera_pos = args.parse_floats("camera_pos", [5.0, 5.0, 2.0])
    camera_look_at = args.parse_floats("camera_look_at", [0.0, 0.0, 1.0])
    camera_settings = {
        'pos': camera_pos,
        'look_at': camera_look_at
    }
    
    assert model_file != "", "请指定 --model_file"
    assert os.path.exists(model_file), f"模型文件不存在: {model_file}"
    
    print(f"加载模型: {model_file}")
    print(f"设备: {device}")
    print(f"环境数量: {num_envs}")
    
    # 构建环境和agent
    env = build_env(args, num_envs, device)
    
    agent_file = args.parse_string("agent_config")
    agent = build_agent(agent_file, env, device)
    
    # 加载模型
    agent.load(model_file)
    print("模型加载成功！")
    
    # 录制视频
    record_video(agent, env, output_video, num_frames, camera_settings)
    
    return


if __name__ == "__main__":
    main(sys.argv)

