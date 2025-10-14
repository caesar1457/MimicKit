# K1 YAML 配置参数详解

## 📊 参数来源依据

所有 K1 的配置参数基于以下三个来源：

### 1. **参考 G1 机器人配置**
K1 和 G1 都是类人型双足机器人，结构相似，因此大部分参数直接参考 G1 的配置。

### 2. **根据 K1_serial.xml 调整**
基于 K1 实际的物理参数进行调整：
- 关节数量：22 DOF（vs G1 的 30 DOF）
- 站立高度：0.65m（vs G1 的 0.8m）
- 质量分布：根据 XML 中的惯性参数

### 3. **经验值和论文推荐**
来自 DeepMimic、AMP、ADD、ASE 论文的推荐值

---

## 🔧 环境配置参数详解 (env.yaml)

### **基础设置**

```yaml
char_file: "data/assets/k1/K1_serial.xml"
```
**依据**: K1 的 MuJoCo 模型文件路径

```yaml
camera_mode: "track"
```
**依据**: G1 配置，"track" 模式让相机跟随机器人移动，更适合观察运动

```yaml
episode_length: 10.0  # seconds
```
**依据**: 
- DeepMimic 论文推荐：5-20秒
- G1 使用 10 秒
- 对于 6.33 秒的运动数据，10 秒足够多次循环

---

### **观测配置**

```yaml
global_obs: True  # DeepMimic/ADD
global_obs: False # AMP/ASE
```
**依据**:
- **DeepMimic/ADD**: 需要全局坐标跟踪参考运动 → `True`
- **AMP/ASE**: 纯模仿学习，局部观测即可 → `False`
- 来源：各算法论文的定义

```yaml
root_height_obs: True
```
**依据**: 所有算法都需要根部高度信息来判断是否摔倒

```yaml
pose_termination: True  # DeepMimic/ADD
pose_termination: False # AMP/ASE
```
**依据**:
- **DeepMimic/ADD**: 偏离参考姿态太远就终止 → 更快收敛
- **AMP/ASE**: 允许探索，不强制终止 → 更多样化

---

### **初始姿态**

```yaml
init_pose: [0, 0, 0.65, 0, 0, 0, ...]  # 28维
```
**依据**:
- **Root position** `[0, 0, 0.65]`:
  - 根据 K1_serial.xml，Trunk 在站立时约 0.65m 高
  - 比 G1 的 0.8m 低（K1 更矮）
  
- **Root rotation** `[0, 0, 0]`:
  - 指数映射表示，[0,0,0] = 无旋转（直立）
  
- **22个关节** 全为 `0`:
  - 所有关节在中位（zero position）
  - 基于 XML 中的 joint range 定义

---

### **终止条件**

```yaml
enable_early_termination: True
termination_height: 0.25
```
**依据**:
- G1 使用 0.3m（站立 0.8m，终止 0.3m ≈ 37.5%）
- K1 站立 0.65m，终止 0.25m ≈ 38.5%（保持相似比例）
- 低于此高度认为摔倒

```yaml
pose_termination_dist: 1.0  # 仅 DeepMimic/ADD
```
**依据**: G1 的值，表示与参考姿态的最大允许距离（米）

---

### **关键Body配置**

```yaml
key_bodies: ["left_foot_link", "right_foot_link", "Head_2", 
             "left_hand_link", "right_hand_link"]
```
**依据**:
- **双脚**: 最重要，用于步态跟踪
- **头部**: 保持平衡和朝向
- **双手**: 手臂摆动协调
- 名称来自 K1_serial.xml 中的 body 定义

```yaml
contact_bodies: ["Left_Shank", "Left_Ankle_Cross", "left_foot_link",
                 "Right_Shank", "Right_Ankle_Cross", "right_foot_link"]
```
**依据**:
- 定义哪些部位允许接触地面
- 小腿、踝关节、脚部 - 正常步态会接触
- 如果其他部位接触（如手、躯干）可能需要惩罚

---

### **关节奖励权重**

```yaml
joint_err_w: [0.5, 0.5,  # 头部(2)
              1.0, 1.0, 1.0, 0.6,  # 左臂(4)
              1.0, 1.0, 1.0, 0.6,  # 右臂(4)
              1.0, 1.0, 1.0, 0.6, 0.5, 0.5,  # 左腿(6)
              1.0, 1.0, 1.0, 0.6, 0.5, 0.5]  # 右腿(6)
```
**依据**:
- **头部 (0.5)**: 不太重要，允许更大误差
- **主要关节 (1.0)**: Hip, Shoulder, Elbow, Knee - 最重要
- **次要关节 (0.6)**: Elbow_Yaw - 不太影响运动质量
- **末端关节 (0.5)**: 踝关节 - 有一定容差
- 参考 G1 的权重分布模式

---

### **奖励权重**

```yaml
reward_pose_w: 0.5
reward_vel_w: 0.1
reward_root_pose_w: 0.15
reward_root_vel_w: 0.1
reward_key_pos_w: 0.15
```
**依据**: 来自 DeepMimic 论文 + G1 调优
- **总和 = 1.0** (0.5+0.1+0.15+0.1+0.15)
- **pose (0.5)**: 最重要，占一半
- **key_pos (0.15)**: 关键点位置很重要
- **root_pose (0.15)**: 保持直立
- **速度 (0.2 总)**: 次要，防止过度关注静态匹配

```yaml
reward_pose_scale: 0.25
reward_vel_scale: 0.01
reward_root_pose_scale: 5.0
reward_root_vel_scale: 1.0
reward_key_pos_scale: 10.0
```
**依据**: DeepMimic 论文推荐 + 实验调优
- **Scale** 控制奖励的敏感度
- 数值较大 = 对误差更敏感
- `key_pos_scale: 10.0` - 手脚位置误差惩罚大
- `root_pose_scale: 5.0` - 倾斜惩罚大
- `vel_scale: 0.01` - 速度误差容忍度高

---

### **判别器配置 (AMP/ADD/ASE)**

```yaml
num_disc_obs_steps: 10  # AMP/ASE
num_disc_obs_steps: 1   # ADD
```
**依据**:
- **AMP/ASE**: 使用过去 10 帧的观测（时序信息）
- **ADD**: 只用当前帧的差分（Differential）
- 来源：各论文定义

```yaml
rand_reset: True
default_reset_prob: 0.5
```
**依据**: AMP/ASE 论文
- 50% 概率从运动数据的随机时刻开始
- 50% 概率正常重置
- 增加训练多样性

---

## 🤖 Agent 配置参数详解 (agent.yaml)

### **网络结构**

```yaml
actor_net: "fc_2layers_1024units"
critic_net: "fc_2layers_1024units"
```
**依据**: G1 的配置
- 2层全连接，每层 1024 个神经元
- 对于 22 DOF 的 K1，这个尺寸足够（G1 是 30 DOF）
- 可以用更小的网络（512 units）节省计算

```yaml
actor_init_output_scale: 0.01
```
**依据**: PPO 论文推荐
- 初始化策略输出接近 0（小随机动作）
- 有助于训练初期稳定性

```yaml
action_std: 0.05
```
**依据**: DeepMimic 论文 + 调优
- 固定的动作标准差（探索噪声）
- 0.05 对位置控制来说是合理的探索幅度
- 太大 → 不稳定；太小 → 探索不足

---

### **优化器**

```yaml
type: "SGD"
learning_rate: 5e-5  # DeepMimic
learning_rate: 1e-4  # AMP/ADD/ASE
```
**依据**:
- **DeepMimic (5e-5)**: 较保守，更稳定
- **AMP/ADD/ASE (1e-4)**: 有判别器帮助，可以更激进
- 来源：各论文的超参数表

---

### **训练参数**

```yaml
discount: 0.99
```
**依据**: RL 标准值（论文推荐）
- γ = 0.99，重视长期回报
- 10秒 episode，30Hz = 300 steps
- 有效时域 ≈ 1/(1-0.99) = 100 steps ≈ 3.3秒

```yaml
steps_per_iter: 32
```
**依据**: G1 配置
- 每次迭代收集 32 步经验
- 4096 envs × 32 steps = 131,072 样本/迭代

```yaml
batch_size: 4
```
**依据**: 经验buffer 大小
- 将 131k 样本分成 4 个 batch 训练
- 每个 batch ≈ 32k 样本

```yaml
update_epochs: 5
```
**依据**: PPO 论文
- 用同一批数据训练 5 轮
- 平衡样本效率 vs 过拟合

```yaml
td_lambda: 0.95
```
**依据**: GAE (Generalized Advantage Estimation) 论文
- λ = 0.95，在偏差-方差间折衷
- 标准 RL 值

```yaml
ppo_clip_ratio: 0.2
```
**依据**: PPO 论文
- ε = 0.2，限制策略更新幅度
- 防止策略崩溃

---

### **判别器参数 (AMP/ADD/ASE)**

```yaml
disc_loss_weight: 5
disc_reward_scale: 2
```
**依据**: AMP 论文
- 判别器损失权重和奖励缩放
- 平衡任务奖励和模仿奖励

```yaml
disc_grad_penalty: 5  # AMP/ASE
disc_grad_penalty: 1  # ADD
```
**依据**:
- **AMP/ASE**: 使用 WGAN-GP，梯度惩罚 5
- **ADD**: 差分判别器，梯度惩罚 1
- 来源：各论文的消融实验

---

### **ASE 特有参数**

```yaml
latent_dim: 64
latent_net: "fc_2layers_512units"
enc_loss_weight: 1.0
```
**依据**: ASE 论文
- 64 维潜在技能编码
- 编码器网络（512 units，比 actor 小）
- 编码器损失权重

---

## 🎯 仿真引擎参数

```yaml
control_mode: "pos"
```
**依据**: K1_serial.xml 定义的是位置控制电机

```yaml
sim_freq: 60
control_freq: 30
```
**依据**:
- **您的数据 FPS = 30** → `control_freq: 30`
- `sim_freq: 60` = 2× 控制频率（标准做法）
- `substeps: 2` → 实际物理模拟 120 Hz

```yaml
env_spacing: 5
```
**依据**: G1 配置
- 环境之间间隔 5 米（避免碰撞）

```yaml
plane:
    static_friction: 1.0
    dynamic_friction: 1.0
    restitution: 0.0
```
**依据**: 
- 摩擦系数 1.0 = 典型地面
- restitution 0 = 不弹跳（稳定接触）

---

## 📊 算法对比总结

| 参数 | DeepMimic | AMP | ADD | ASE |
|------|-----------|-----|-----|-----|
| **global_obs** | ✅ | ❌ | ✅ | ❌ |
| **pose_termination** | ✅ | ❌ | ✅ | ❌ |
| **learning_rate** | 5e-5 | 1e-4 | 1e-4 | 1e-4 |
| **num_disc_obs_steps** | - | 10 | 1 | 10 |
| **disc_grad_penalty** | - | 5 | 1 | 5 |
| **特殊组件** | - | Discriminator | Diff Disc | Encoder |

---

## 🔬 如何调整参数

### 机器人摔倒太多？
1. ↓ `termination_height: 0.2`
2. ↑ `reward_root_pose_w: 0.2`
3. ↓ `action_std: 0.03`

### 学习太慢？
1. ↑ `learning_rate: 2e-4`
2. ↑ `batch_size: 8`
3. ↑ `num_envs: 8192`

### 动作不平滑？
1. ↑ `action_reg_weight: 0.01`
2. ↓ `action_std: 0.03`

### 模仿质量不够？
1. ↑ `reward_pose_w: 0.6`
2. ↑ `disc_loss_weight: 10`
3. ↑ `reward_key_pos_scale: 15.0`

---

## 📚 参考文献

1. **DeepMimic**: Peng et al., "DeepMimic: Example-guided Deep RL", TOG 2018
2. **AMP**: Peng et al., "AMP: Adversarial Motion Priors", TOG 2021
3. **ASE**: Peng et al., "ASE: Large-scale Adversarial Skill Embeddings", TOG 2022
4. **ADD**: Zhang et al., "Physics-Based Motion Imitation with Adversarial Differential Discriminators", SIGGRAPH Asia 2025
5. **G1 配置**: MimicKit 官方示例
6. **PPO**: Schulman et al., "Proximal Policy Optimization", 2017

---

**总结**: 所有参数都有明确的来源依据，主要基于：
- ✅ 学术论文的推荐值
- ✅ G1 机器人的成功配置
- ✅ K1 实际物理参数
- ✅ 强化学习的标准实践


