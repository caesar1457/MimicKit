#!/usr/bin/env python
import torch
import numpy as np
import sys
sys.path.insert(0, 'mimickit')

# 简单测试 K1 环境是否能正常初始化和运行第一步

import envs.env_builder as env_builder

device = torch.device("cuda:0")
num_envs = 4  # 只用 4 个环境测试

print("Building environment...")
env = env_builder.build_env("data/envs/add_k1_env.yaml", num_envs, device, visualize=False)

print(f"Environment built successfully!")
print(f"Observation shape: {env.get_obs_size()}")
print(f"Action shape: {env.get_action_size()}")

# 重置环境
print("\nResetting environment...")
obs, info = env.reset()

print(f"Initial observation shape: {obs.shape}")
print(f"Has NaN in observation: {torch.isnan(obs).any().item()}")
print(f"Has Inf in observation: {torch.isinf(obs).any().item()}")
print(f"Observation stats: min={obs.min().item():.4f}, max={obs.max().item():.4f}, mean={obs.mean().item():.4f}")

# 尝试执行一步
print("\nExecuting one step with zero actions...")
actions = torch.zeros(num_envs, env.get_action_size(), device=device)
obs_next, reward, done, info_next = env.step(actions)

print(f"After step:")
print(f"  Has NaN in observation: {torch.isnan(obs_next).any().item()}")
print(f"  Has Inf in observation: {torch.isinf(obs_next).any().item()}")
print(f"  Has NaN in reward: {torch.isnan(reward).any().item()}")
print(f"  Has Inf in reward: {torch.isinf(reward).any().item()}")
print(f"  Observation stats: min={obs_next.min().item():.4f}, max={obs_next.max().item():.4f}")
print(f"  Reward stats: min={reward.min().item():.4f}, max={reward.max().item():.4f}, mean={reward.mean().item():.4f}")

print("\n✓ Environment test completed successfully!")

