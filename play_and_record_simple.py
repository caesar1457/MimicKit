"""
简化版：Play训练好的模型并录制视频（适合服务器）

用法示例：
CUDA_VISIBLE_DEVICES=6 python play_and_record_simple.py \
    --arg_file args/add_k1_args.txt \
    --model_file output/k1_add/k1_add_model.pt \
    --output_dir output/k1_video \
    --num_episodes 3 \
    --max_steps 500

然后使用ffmpeg合成视频：
ffmpeg -framerate 30 -i output/k1_video/frame_%06d.png -c:v libx264 -pix_fmt yuv420p output/k1_demo.mp4
"""

import numpy as np
import os
import sys
import time
import torch

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.arg_parser as arg_parser
from util.logger import Logger


def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])

    arg_file = args.parse_string("arg_file", "")
    if arg_file != "":
        succ = args.load_file(arg_file)
        assert succ, Logger.print("Failed to load args from: " + arg_file)

    return args


def build_env(args, num_envs, device, visualize):
    env_file = args.parse_string("env_config")
    env = env_builder.build_env(env_file, num_envs, device, visualize)
    return env


def build_agent(agent_file, env, device):
    agent = agent_builder.build_agent(agent_file, env, device)
    return agent


def test_and_record(agent, env, output_dir, num_episodes=3, max_steps=500, record_interval=1):
    """
    测试agent并保存帧图像
    
    Args:
        agent: Trained agent
        env: Environment
        output_dir: 图像保存目录
        num_episodes: 要录制的episode数量
        max_steps: 每个episode的最大步数
        record_interval: 每隔多少步保存一帧（1=每步都保存）
    """
    print(f"开始测试并录制，共 {num_episodes} 个episodes，每个最多 {max_steps} 步")
    print(f"图像保存到: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取gym和viewer
    gym = env._engine._gym
    sim = env._engine._sim
    viewer = env._engine._viewer
    
    frame_count = 0
    episode_count = 0
    total_return = 0
    
    # 重置环境
    obs, info = env.reset()
    episode_return = 0
    episode_steps = 0
    
    while episode_count < num_episodes:
        # 获取动作
        with torch.no_grad():
            action_info = agent.decide_action(obs, info)
            action = action_info["action"]
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        episode_return += reward[0].item()
        episode_steps += 1
        
        # 保存帧（每record_interval步）
        if frame_count % record_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.png")
            try:
                # 尝试写入图像
                gym.write_viewer_image_to_file(viewer, frame_path)
            except:
                # 如果失败，打印提示
                if frame_count == 0:
                    print("警告: 无法保存图像。请确保运行在有显示的环境中。")
                    print("对于无头服务器，建议使用camera sensor或EGL渲染。")
        
        frame_count += 1
        
        # 检查是否完成或达到最大步数
        if done[0] or episode_steps >= max_steps:
            episode_count += 1
            total_return += episode_return
            
            print(f"Episode {episode_count}/{num_episodes} 完成:")
            print(f"  - 步数: {episode_steps}")
            print(f"  - 回报: {episode_return:.2f}")
            print(f"  - 累计帧数: {frame_count}")
            
            if episode_count < num_episodes:
                # 重置环境继续
                obs, info = env.reset()
                episode_return = 0
                episode_steps = 0
    
    avg_return = total_return / num_episodes
    print(f"\n测试完成!")
    print(f"平均回报: {avg_return:.2f}")
    print(f"总帧数: {frame_count}")
    print(f"\n要生成视频，请运行:")
    print(f"ffmpeg -framerate 30 -i {output_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p {output_dir}/video.mp4")
    
    return avg_return


def main(argv):
    args = load_args(argv)
    
    # 解析参数
    device = args.parse_string("device", "cuda:0")
    num_envs = args.parse_int("num_envs", 1)  # 录制时建议用1个环境
    model_file = args.parse_string("model_file", "")
    output_dir = args.parse_string("output_dir", "output/video_frames")
    num_episodes = args.parse_int("num_episodes", 3)
    max_steps = args.parse_int("max_steps", 500)
    record_interval = args.parse_int("record_interval", 1)
    
    assert model_file != "", "请指定 --model_file 参数"
    assert os.path.exists(model_file), f"模型文件不存在: {model_file}"
    
    print("="*60)
    print(f"模型文件: {model_file}")
    print(f"设备: {device}")
    print(f"环境数量: {num_envs}")
    print(f"输出目录: {output_dir}")
    print("="*60)
    
    # 构建环境和agent（开启可视化）
    env = build_env(args, num_envs, device, visualize=True)
    
    agent_file = args.parse_string("agent_config")
    agent = build_agent(agent_file, env, device)
    
    # 加载模型
    agent.load(model_file)
    print("✓ 模型加载成功！\n")
    
    # 测试并录制
    avg_return = test_and_record(agent, env, output_dir, num_episodes, max_steps, record_interval)
    
    return


if __name__ == "__main__":
    main(sys.argv)

