#!/usr/bin/env python
"""
测试 K1 模型 play 是否能正常站立
"""
import torch
import sys
sys.path.insert(0, 'mimickit')

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder

print("=" * 60)
print("K1 Play 测试 - 检查初始化是否倒地")
print("=" * 60)

device = torch.device("cuda:0")
num_envs = 4

# 1. 构建环境
print("\n1. 构建环境...")
env = env_builder.build_env("data/envs/add_k1_env.yaml", num_envs, device, visualize=True)
print(f"✓ 环境构建成功")

# 2. 重置环境
print("\n2. 重置环境并检查初始状态...")
obs, info = env.reset()

# 获取机器人的初始状态
if hasattr(env, '_engine'):
    root_states = env._engine.get_root_states()
    print(f"\n初始 Root 状态:")
    print(f"  位置: {root_states[0, :3].cpu().numpy()}")
    print(f"  高度: {root_states[0, 2].item():.4f} m")
    print(f"  旋转 (quat): {root_states[0, 3:7].cpu().numpy()}")

# 3. 执行若干步，观察是否倒地
print("\n3. 执行 100 步，观察是否倒地...")
for step in range(100):
    # 使用零动作（机器人应该尽量保持姿态）
    actions = torch.zeros(num_envs, env.get_action_size(), device=device)
    obs, reward, done, info = env.step(actions)
    
    if step % 20 == 0:
        root_states = env._engine.get_root_states()
        height = root_states[0, 2].item()
        print(f"  Step {step:3d}: 高度 = {height:.4f} m, 完成 = {done[0].item()}")
        
        # 检查是否倒地（高度低于阈值）
        if height < 0.3:
            print(f"\n❌ 警告：机器人可能已倒地！(高度 < 0.3m)")
            break
else:
    print(f"\n✅ 测试通过！机器人在 100 步内保持站立")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)




