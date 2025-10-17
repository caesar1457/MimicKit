#!/usr/bin/env python
import sys
sys.path.insert(0, 'mimickit')
import torch
import numpy as np

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder

print("=" * 60)
print("K1 NaN 调试脚本")
print("=" * 60)

device = torch.device("cuda:0")
num_envs = 4

print("\n1. 构建环境...")
env = env_builder.build_env("data/envs/add_k1_env.yaml", num_envs, device, visualize=False)
print(f"✓ 环境构建成功")

print("\n2. 重置环境...")
obs, info = env.reset()
print(f"✓ 环境重置成功")
print(f"  Observation shape: {obs.shape}")
print(f"  Observation 统计:")
print(f"    min={obs.min().item():.4f}, max={obs.max().item():.4f}, mean={obs.mean().item():.4f}")
print(f"    has NaN: {torch.isnan(obs).any().item()}")
print(f"    has Inf: {torch.isinf(obs).any().item()}")

print("\n3. 构建 Agent...")
agent = agent_builder.build_agent("data/agents/add_k1_agent.yaml", env, device)
print(f"✓ Agent 构建成功")

print("\n4. 获取初始 action...")
with torch.no_grad():
    action, action_info = agent.decide_action(obs, info)
print(f"✓ Action 生成成功")
print(f"  Action shape: {action.shape}")
print(f"  Action 统计:")
print(f"    min={action.min().item():.4f}, max={action.max().item():.4f}, mean={action.mean().item():.4f}")
print(f"    has NaN: {torch.isnan(action).any().item()}")
print(f"    has Inf: {torch.isinf(action).any().item()}")

print("\n5. 执行一步环境...")
obs_next, reward, done, info_next = env.step(action)
print(f"✓ 环境 step 成功")
print(f"  Reward 统计:")
print(f"    min={reward.min().item():.4f}, max={reward.max().item():.4f}, mean={reward.mean().item():.4f}")
print(f"    has NaN: {torch.isnan(reward).any().item()}")
print(f"    has Inf: {torch.isinf(reward).any().item()}")
print(f"  Next obs 统计:")
print(f"    min={obs_next.min().item():.4f}, max={obs_next.max().item():.4f}, mean={obs_next.mean().item():.4f}")
print(f"    has NaN: {torch.isnan(obs_next).any().item()}")
print(f"    has Inf: {torch.isinf(obs_next).any().item()}")

print("\n6. 测试 Normalizer...")
# 模拟收集一些数据
agent.set_mode(agent_builder.base_agent.AgentMode.TRAIN)
for i in range(10):
    with torch.no_grad():
        action, action_info = agent.decide_action(obs, info)
    obs_next, reward, done, info_next = env.step(action)
    
    # 检查 normalizer 的状态
    if hasattr(agent, '_obs_norm'):
        obs_mean = agent._obs_norm.get_mean()
        obs_std = agent._obs_norm.get_std()
        print(f"\n  Step {i}: Normalizer stats:")
        print(f"    obs_mean: min={obs_mean.min().item():.6f}, max={obs_mean.max().item():.6f}")
        print(f"    obs_std: min={obs_std.min().item():.6f}, max={obs_std.max().item():.6f}")
        print(f"    obs_std has near-zero: {(obs_std < 1e-6).any().item()}")
        
        if (obs_std < 1e-6).any():
            print(f"    WARNING: obs_std 有接近0的值!")
            near_zero_idx = (obs_std < 1e-6).nonzero(as_tuple=True)[0]
            print(f"    Near-zero indices: {near_zero_idx[:10].tolist()}")
    
    obs = obs_next
    info = info_next
    
    if i == 0:
        break

print("\n7. 检查 motion 数据处理...")
# 获取 motion lib
if hasattr(env, '_motion_lib'):
    motion_lib = env._motion_lib
    print(f"  Motion lib 信息:")
    print(f"    Num motions: {motion_lib.get_num_motions()}")
    print(f"    Total length: {motion_lib.get_total_length():.3f}s")
    
    # 采样一些 motion 数据
    motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
    motion_times = torch.zeros(num_envs, device=device)
    
    root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = motion_lib.get_motion_state(motion_ids, motion_times)
    
    print(f"\n  Motion state 统计:")
    print(f"    root_pos: min={root_pos.min().item():.4f}, max={root_pos.max().item():.4f}, has NaN={torch.isnan(root_pos).any().item()}")
    print(f"    root_rot: min={root_rot.min().item():.4f}, max={root_rot.max().item():.4f}, has NaN={torch.isnan(root_rot).any().item()}")
    print(f"    dof_pos: min={dof_pos.min().item():.4f}, max={dof_pos.max().item():.4f}, has NaN={torch.isnan(dof_pos).any().item()}")
    print(f"    dof_vel: min={dof_vel.min().item():.4f}, max={dof_vel.max().item():.4f}, has NaN={torch.isnan(dof_vel).any().item()}")

print("\n" + "=" * 60)
print("调试完成")
print("=" * 60)

