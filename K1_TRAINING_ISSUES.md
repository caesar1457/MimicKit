# K1 训练问题诊断报告

## 问题现状

K1 机器人在 ADD 训练中第一次迭代就出现 NaN，无法正常训练。

### 症状

```
Iteration 0:
- Critic_Loss: 36.3 (非常高，正常应该 < 5)
- Actor_Loss: nan
- 之后所有 loss 都变成 nan
```

### 已完成的修复

#### 1. Motion 数据加载 ✓
**文件**: `mimickit/anim/motion.py`
- 修复 numpy 版本兼容性 (`numpy._core` → `numpy.core`)
- 修复 `__main__.Motion` 类映射问题

#### 2. 训练参数优化 ✓
**文件**: `data/agents/add_k1_agent.yaml`
```yaml
learning_rate: 2e-5        # 降低5倍防止梯度爆炸
grad_clip: 10.0            # 添加大幅度梯度裁剪
batch_size: 8              # 增加批次大小提高稳定性
update_epochs: 3           # 减少更新次数
iters_per_output: 10       # 更频繁输出便于调试
```

#### 3. 环境配置 ✓
**文件**: `data/envs/add_k1_env.yaml`
```yaml
# 物理引擎缓冲区
max_gpu_contact_pairs: 8388608
found_lost_pairs_capacity: 524288

# 初始姿态（简单站立）
init_pose: [0, 0, 0.65, 0, 0, 0, ...]  # 28个值
```

#### 4. 运行参数 ✓
**文件**: `args/add_k1_args.txt`
```
--num_envs 64             # 从4096降到64
--logger tb               # 添加TensorBoard日志
```

### 根本问题分析

经过系统性测试，发现：

1. **环境数量影响**
   - 256/512 envs: 1300万碰撞对 → 物理引擎崩溃
   - 64 envs: 无碰撞警告 → 但仍然 NaN
   - 说明：碰撞问题已解决，但 NaN 是训练参数问题

2. **Critic Loss 异常高**
   - Critic_Loss = 36.3（正常应 < 5）
   - 说明：状态值估计严重偏差

3. **第一次迭代就 NaN**
   - 不是逐渐变 NaN，而是立即出现
   - 说明：初始化或第一次梯度更新就有问题

### 可能的原因

1. **Observation 归一化问题**
   - K1 的观测空间可能与 G1 不同
   - Normalizer 在第一次迭代可能除以接近0的std

2. **Reward Scale 不匹配**
   - disc_reward_scale: 2.0 可能对 K1 太大
   - Advantage 值：4.74（可能偏大）

3. **Motion 数据与模型不完全匹配**
   - Motion 数据虽然能加载，但可能不是针对当前 K1 模型录制的
   - 导致判别器和策略网络的输入异常

### 建议的下一步

#### 方案 1：对比测试（推荐）
```bash
# 先确认 G1 能正常训练
python mimickit/run.py --mode train --arg_file args/add_g1_args.txt --num_envs 64
```
如果 G1 正常，说明是 K1 特定问题。

#### 方案 2：极端保守参数
修改 `data/agents/add_k1_agent.yaml`:
```yaml
learning_rate: 5e-6        # 极低学习率
disc_reward_scale: 0.2     # 大幅降低判别器奖励
batch_size: 16             # 更大批次
```

#### 方案 3：检查 Motion 数据来源
- 这个 pkl 文件是如何生成的？
- 是否与当前 K1 模型的关节配置完全匹配？
- 建议联系数据提供者确认兼容性

#### 方案 4：调试模式
添加调试输出查看具体哪个tensor变成NaN：
```python
# 在 mimickit/learning/add_agent.py 的 _compute_loss 中添加
if torch.isnan(loss).any():
    print(f"NaN in loss components:")
    print(f"  actor_loss: {actor_loss}")
    print(f"  critic_loss: {critic_loss}")
    print(f"  disc_loss: {disc_loss}")
```

### 当前配置摘要

**K1 模型信息**
- DoFs: 22
- Mass: 18.714 kg
- Motion: 1260 frames, 30 fps, 41.967s

**训练配置**
- Environments: 64
- Learning rate: 2e-5
- Batch size: 8
- Gradient clip: 10.0

### 结论

K1 的训练问题不是简单的参数调整能解决的。最可能的原因是：
1. Motion 数据与机器人模型不匹配
2. Observation/Reward scale 需要针对 K1 重新调整
3. 可能需要 K1 专用的训练配置

建议先用 G1 验证代码无问题，然后联系 motion 数据提供者确认数据兼容性。

