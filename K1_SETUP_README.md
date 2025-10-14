# K1 机器人 MimicKit 配置说明

## 📁 已创建的配置文件

### 环境配置 (data/envs/)
- ✅ `deepmimic_k1_env.yaml` - DeepMimic 环境配置
- ✅ `add_k1_env.yaml` - ADD (Adversarial Differential Discriminators) 环境配置
- ✅ `amp_k1_env.yaml` - AMP (Adversarial Motion Priors) 环境配置
- ✅ `view_motion_k1_env.yaml` - 运动可视化环境配置

### Agent 配置 (data/agents/)
- ✅ `deepmimic_k1_ppo_agent.yaml` - DeepMimic PPO 智能体
- ✅ `add_k1_agent.yaml` - ADD 智能体
- ✅ `amp_k1_agent.yaml` - AMP 智能体

### 参数文件 (args/)
- ✅ `deepmimic_k1_ppo_args.txt` - DeepMimic 训练参数
- ✅ `add_k1_args.txt` - ADD 训练参数
- ✅ `amp_k1_args.txt` - AMP 训练参数
- ✅ `view_motion_k1_args.txt` - 运动查看参数

## 🤖 K1 机器人配置详情

### 关节配置（22个DOF）
```
头部 (2):  Head_yaw, Head_pitch
左臂 (4):  Left_Shoulder_Pitch, Left_Shoulder_Roll, Left_Elbow_Pitch, Left_Elbow_Yaw
右臂 (4):  Right_Shoulder_Pitch, Right_Shoulder_Roll, Right_Elbow_Pitch, Right_Elbow_Yaw
左腿 (6):  Left_Hip_Pitch, Left_Hip_Roll, Left_Hip_Yaw, Left_Knee_Pitch, Left_Ankle_Pitch, Left_Ankle_Roll
右腿 (6):  Right_Hip_Pitch, Right_Hip_Roll, Right_Hip_Yaw, Right_Knee_Pitch, Right_Ankle_Pitch, Right_Ankle_Roll
```

### 关键Body
- `left_foot_link`, `right_foot_link` - 双脚（用于奖励跟踪）
- `Head_2` - 头部
- `left_hand_link`, `right_hand_link` - 双手

### 接触Body
- `Left_Shank`, `Right_Shank` - 小腿
- `Left_Ankle_Cross`, `Right_Ankle_Cross` - 踝关节
- `left_foot_link`, `right_foot_link` - 脚部

### 初始姿态
- 站立高度：0.65m（Trunk 中心）
- 所有关节初始角度：0（中位）

## 📦 准备运动数据

### ⚠️ 重要：您需要准备运动数据文件

运动数据应该放在：
```
data/motions/k1/k1_walk.pkl
```

### 运动数据格式

每一帧的数据格式：
```python
frame = [
    root_pos (3D),      # [x, y, z]
    root_rot (3D),      # [rx, ry, rz] 指数映射表示
    joint_angles (22D)  # 按照XML中的关节顺序
]
# 总维度：3 + 3 + 22 = 28
```

### 关节顺序（必须严格遵守）
```python
joint_order = [
    "Head_yaw",              # 0
    "Head_pitch",            # 1
    "Left_Shoulder_Pitch",   # 2
    "Left_Shoulder_Roll",    # 3
    "Left_Elbow_Pitch",      # 4
    "Left_Elbow_Yaw",        # 5
    "Right_Shoulder_Pitch",  # 6
    "Right_Shoulder_Roll",   # 7
    "Right_Elbow_Pitch",     # 8
    "Right_Elbow_Yaw",       # 9
    "Left_Hip_Pitch",        # 10
    "Left_Hip_Roll",         # 11
    "Left_Hip_Yaw",          # 12
    "Left_Knee_Pitch",       # 13
    "Left_Ankle_Pitch",      # 14
    "Left_Ankle_Roll",       # 15
    "Right_Hip_Pitch",       # 16
    "Right_Hip_Roll",        # 17
    "Right_Hip_Yaw",         # 18
    "Right_Knee_Pitch",      # 19
    "Right_Ankle_Pitch",     # 20
    "Right_Ankle_Roll",      # 21
]
```

## 🚀 使用方法

### 1. 查看运动数据（推荐先做这个）

```bash
python mimickit/run.py --arg_file args/view_motion_k1_args.txt --visualize true
```

这会播放您的运动数据，帮助验证：
- ✅ 运动数据格式正确
- ✅ 关节顺序正确
- ✅ K1 模型正确加载

### 2. 训练 DeepMimic + PPO

```bash
# 训练（关闭可视化以加速）
python mimickit/run.py --arg_file args/deepmimic_k1_ppo_args.txt --visualize false

# 或者用更少的环境数测试
python mimickit/run.py --arg_file args/deepmimic_k1_ppo_args.txt --num_envs 512 --visualize true
```

### 3. 训练 ADD（推荐用于复杂动作）

```bash
python mimickit/run.py --arg_file args/add_k1_args.txt --visualize false
```

### 4. 训练 AMP（纯模仿学习）

```bash
python mimickit/run.py --arg_file args/amp_k1_args.txt --visualize false
```

### 5. 测试训练好的模型

```bash
python mimickit/run.py \
    --arg_file args/deepmimic_k1_ppo_args.txt \
    --mode test \
    --num_envs 4 \
    --visualize true \
    --model_file output/k1_deepmimic_ppo/model.pt
```

## 📊 监控训练

### TensorBoard
```bash
tensorboard --logdir=output/ --port=6006 --bind_all
```

### Wandb
```bash
# 在 args 文件中添加
--logger wandb
```

## 🔧 调优建议

### 1. 如果机器人摔倒太多
```yaml
# 在 env 配置中调整
termination_height: 0.2  # 降低终止高度阈值
reward_root_pose_w: 0.2  # 增大根姿态奖励
```

### 2. 如果动作不够平滑
```yaml
# 在 agent 配置中调整
action_std: 0.1  # 增大探索噪声（初期）
action_reg_weight: 0.01  # 增加动作正则化
```

### 3. 如果学习太慢
```yaml
# 在 agent 配置中调整
learning_rate: 1e-4  # 增大学习率
batch_size: 8  # 增大批量大小
```

### 4. 如果需要更好的模仿
```yaml
# 在 env 配置中调整
reward_pose_w: 0.7  # 增大姿态奖励权重
reward_key_pos_w: 0.2  # 增大关键点跟踪权重
```

## 📝 配置参数说明

### 环境参数
- `episode_length`: 每个 episode 的长度（秒）
- `global_obs`: 是否使用全局坐标系观测
- `termination_height`: 低于此高度会终止 episode
- `joint_err_w`: 每个关节的奖励权重（共22个值）

### 奖励参数
- `reward_pose_w`: 姿态跟踪奖励权重
- `reward_vel_w`: 速度跟踪奖励权重
- `reward_root_pose_w`: 根姿态奖励权重
- `reward_key_pos_w`: 关键点位置奖励权重

### 训练参数
- `num_envs`: 并行环境数量（建议4096）
- `steps_per_iter`: 每次迭代的步数
- `iters_per_output`: 每隔多少次迭代输出一次

## ⚠️ 常见问题

### Q1: 提示找不到运动文件
```
FileNotFoundError: data/motions/k1/k1_walk.pkl
```
**解决**：您需要先准备运动数据文件，放到 `data/motions/k1/` 目录下

### Q2: 关节数量不匹配
```
AssertionError: Joint count mismatch
```
**解决**：检查您的运动数据关节顺序是否与 K1_serial.xml 一致

### Q3: Body 名称找不到
```
RuntimeError: Body 'xxx' not found
```
**解决**：检查环境配置中的 `key_bodies` 和 `contact_bodies` 名称是否与 XML 文件一致

### Q4: init_pose 维度不对
```
AssertionError: init_pose dimension mismatch
```
**解决**：init_pose 应该是 28 维：[root_pos(3), root_rot(3), joints(22)]

## 🎯 下一步

1. ✅ 配置文件已创建
2. ⏳ **准备运动数据** - 将 `.pkl` 文件放到 `data/motions/k1/`
3. ⏳ 使用 `view_motion` 验证数据
4. ⏳ 开始训练
5. ⏳ 调优参数

## 📚 参考资料

- DeepMimic 论文: https://arxiv.org/abs/1804.02717
- AMP 论文: https://arxiv.org/abs/2104.02180
- ADD 论文: https://arxiv.org/abs/2410.20324


