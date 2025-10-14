# K1 快速开始指南 🚀

## ✅ 已完成的配置

所有 K1 机器人的配置文件已创建并通过验证！

### 📁 创建的文件清单

```
✅ 环境配置 (4个)
   ├── data/envs/deepmimic_k1_env.yaml
   ├── data/envs/add_k1_env.yaml
   ├── data/envs/amp_k1_env.yaml
   └── data/envs/view_motion_k1_env.yaml

✅ Agent配置 (3个)
   ├── data/agents/deepmimic_k1_ppo_agent.yaml
   ├── data/agents/add_k1_agent.yaml
   └── data/agents/amp_k1_agent.yaml

✅ 参数文件 (4个)
   ├── args/deepmimic_k1_ppo_args.txt
   ├── args/add_k1_args.txt
   ├── args/amp_k1_args.txt
   └── args/view_motion_k1_args.txt

✅ 辅助工具 (2个)
   ├── tools/create_k1_motion_template.py
   └── tools/verify_k1_config.py

✅ 文档
   ├── K1_SETUP_README.md (详细文档)
   └── K1_QUICKSTART.md (本文件)
```

## 🎯 立即开始（3步）

### 步骤 1: 创建测试运动数据

如果您已经有运动数据，跳过这步。否则创建一个测试用的：

```bash
python tools/create_k1_motion_template.py
```

这会创建一个简单的步行运动在 `data/motions/k1/k1_walk.pkl`

### 步骤 2: 验证运动数据

查看运动是否正确：

```bash
python mimickit/run.py --arg_file args/view_motion_k1_args.txt --visualize true
```

您应该看到 K1 机器人在执行步行动作。

### 步骤 3: 开始训练

选择一个算法开始训练：

**DeepMimic (推荐初学者):**
```bash
python mimickit/run.py --arg_file args/deepmimic_k1_ppo_args.txt --visualize false
```

**ADD (更好的模仿质量):**
```bash
python mimickit/run.py --arg_file args/add_k1_args.txt --visualize false
```

**AMP (纯模仿学习):**
```bash
python mimickit/run.py --arg_file args/amp_k1_args.txt --visualize false
```

## 📊 监控训练

### TensorBoard
```bash
# 新开一个终端
tensorboard --logdir=output/ --port=6006 --bind_all
# 然后打开浏览器访问 http://localhost:6006
```

### 查看日志
```bash
tail -f output/k1_deepmimic_ppo/log.txt
```

## 🧪 测试训练好的模型

训练完成后（或中途）测试模型：

```bash
python mimickit/run.py \
    --arg_file args/deepmimic_k1_ppo_args.txt \
    --mode test \
    --num_envs 4 \
    --visualize true \
    --model_file output/k1_deepmimic_ppo/model.pt
```

## 🎛️ 常用调整

### 1. 减少环境数量（快速测试）

```bash
python mimickit/run.py \
    --arg_file args/deepmimic_k1_ppo_args.txt \
    --num_envs 512 \
    --visualize true
```

### 2. 从检查点继续训练

```bash
python mimickit/run.py \
    --arg_file args/deepmimic_k1_ppo_args.txt \
    --model_file output/k1_deepmimic_ppo/model_010000.pt
```

### 3. 修改训练参数

编辑对应的配置文件：
- 环境设置: `data/envs/deepmimic_k1_env.yaml`
- Agent设置: `data/agents/deepmimic_k1_ppo_agent.yaml`

## 📈 期望的训练结果

根据 G1 机器人的经验，K1 的训练应该看到：

| 指标 | 初始值 | 训练中 | 收敛后 |
|------|--------|--------|--------|
| Test_Return | < 0 | 逐渐增加 | > 2.0 |
| Episode_Length | 短 | 逐渐增加 | 接近 episode_length |
| Pose_Error | 大 | 逐渐减小 | < 0.1 |

训练时长（4096个环境）：
- **DeepMimic**: 约 4-8 小时（取决于GPU）
- **ADD/AMP**: 约 8-12 小时

## ⚠️ 故障排除

### 问题1: 机器人立即摔倒
**解决方案**: 检查 `init_pose` 是否正确，或降低 `termination_height`

```yaml
# 在 env 配置中
termination_height: 0.2  # 降低
```

### 问题2: 运动不够平滑
**解决方案**: 增加动作正则化

```yaml
# 在 agent 配置中
action_reg_weight: 0.01
```

### 问题3: 学习太慢
**解决方案**: 增大学习率或批量大小

```yaml
# 在 agent 配置中
learning_rate: 1e-4  # 增大
batch_size: 8        # 增大
```

### 问题4: CUDA 内存不足
**解决方案**: 减少环境数量

```bash
--num_envs 2048  # 或更小
```

## 📚 K1 机器人规格

- **自由度**: 22 DOF
  - 头部: 2
  - 手臂: 8 (每侧4个)
  - 腿部: 12 (每侧6个)
- **站立高度**: ~0.65m (Trunk中心)
- **控制模式**: 位置控制 (pos)
- **控制频率**: 30 Hz
- **仿真频率**: 60 Hz

## 🎓 算法选择建议

| 算法 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| **DeepMimic** | 简单动作、快速原型 | 简单、快速 | 模仿质量中等 |
| **AMP** | 纯模仿、风格化动作 | 模仿质量好 | 需要更多数据 |
| **ADD** | 复杂动作、高保真 | 最佳模仿质量 | 训练最慢 |

## 🔄 使用真实运动数据

如果您有从真实 K1 或其他来源的运动数据：

### 1. 数据格式要求

```python
# 每帧必须是 28 维向量
frame = [
    root_x, root_y, root_z,           # 根位置 (3)
    root_rx, root_ry, root_rz,        # 根旋转(指数映射) (3)
    # 22个关节角度（按照下面的顺序）
    Head_yaw, Head_pitch,
    L_Shoulder_Pitch, L_Shoulder_Roll, L_Elbow_Pitch, L_Elbow_Yaw,
    R_Shoulder_Pitch, R_Shoulder_Roll, R_Elbow_Pitch, R_Elbow_Yaw,
    L_Hip_Pitch, L_Hip_Roll, L_Hip_Yaw, L_Knee_Pitch, L_Ankle_Pitch, L_Ankle_Roll,
    R_Hip_Pitch, R_Hip_Roll, R_Hip_Yaw, R_Knee_Pitch, R_Ankle_Pitch, R_Ankle_Roll
]
```

### 2. 转换脚本示例

```python
from mimickit.anim.motion import Motion
import numpy as np
import pickle

# 假设您有原始数据
your_data = load_your_motion_data()  # 形状: (num_frames, 28)

# 创建 Motion 对象
motion = Motion(
    frames=your_data, 
    fps=60,           # 您的数据帧率
    loop_mode="wrap"  # "wrap" 或 "none"
)

# 保存
with open('data/motions/k1/your_motion.pkl', 'wb') as f:
    pickle.dump(motion, f)
```

## 🎯 下一步学习

1. 阅读详细文档: `K1_SETUP_README.md`
2. 查看 MimicKit 主 README: `README.md`
3. 研究相关论文:
   - [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/index.html)
   - [AMP](https://xbpeng.github.io/projects/AMP/index.html)
   - [ADD](https://xbpeng.github.io/projects/ADD/index.html)

## 💬 需要帮助？

- 检查配置: `python tools/verify_k1_config.py`
- 查看详细文档: `K1_SETUP_README.md`
- MimicKit 主页: [GitHub](https://github.com/xbpeng/MimicKit)

---

**祝训练顺利！** 🎉


