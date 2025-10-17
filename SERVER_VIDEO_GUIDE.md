# 在服务器上生成K1机器人视频的指南 🎥

## 方法1: 使用虚拟显示器（推荐，最简单）⭐

### 步骤 1: 安装Xvfb

```bash
sudo apt-get install xvfb
```

### 步骤 2: 使用xvfb-run运行测试

```bash
# 设置虚拟显示器并运行
CUDA_VISIBLE_DEVICES=6 xvfb-run -a -s "-screen 0 1280x720x24" \
python mimickit/run.py \
    --arg_file args/add_k1_args.txt \
    --mode test \
    --num_envs 1 \
    --visualize true \
    --model_file output/k1_add/k1_add_model.pt \
    --test_episodes 3
```

### 步骤 3: 录制屏幕（使用ffmpeg）

创建一个录制脚本 `record_test.sh`:

```bash
#!/bin/bash

# 启动虚拟显示器
Xvfb :99 -screen 0 1280x720x24 &
XVFB_PID=$!
export DISPLAY=:99

# 等待X server启动
sleep 2

# 开始录制（后台）
ffmpeg -f x11grab -video_size 1280x720 -i :99 -codec:v libx264 -r 30 output/k1_demo.mp4 &
FFMPEG_PID=$!

# 运行测试
CUDA_VISIBLE_DEVICES=6 python mimickit/run.py \
    --arg_file args/add_k1_args.txt \
    --mode test \
    --num_envs 1 \
    --visualize true \
    --model_file output/k1_add/k1_add_model.pt \
    --test_episodes 3

# 停止录制
kill $FFMPEG_PID
kill $XVFB_PID

echo "视频已保存到 output/k1_demo.mp4"
```

运行：
```bash
chmod +x record_test.sh
./record_test.sh
```

---

## 方法2: 使用Isaac Gym的EGL渲染（无需X server）

### 步骤 1: 修改环境配置以使用EGL

编辑 `data/envs/add_k1_env.yaml`，在 engine 部分添加：

```yaml
engine:
  engine_name: isaac_gym
  sim_freq: 60
  control_freq: 30
  env_spacing: 5
  graphics_device_id: 0  # 添加这行
  use_gpu: true
  use_gpu_pipeline: true
```

### 步骤 2: 设置EGL环境变量

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

### 步骤 3: 运行（这个方法可能需要特殊的Isaac Gym设置）

```bash
CUDA_VISIBLE_DEVICES=6 python mimickit/run.py \
    --arg_file args/add_k1_args.txt \
    --mode test \
    --num_envs 1 \
    --visualize true \
    --model_file output/k1_add/k1_add_model.pt
```

---

## 方法3: 本地可视化（将模型下载到本地）

如果服务器录制困难，可以把模型下载到本地电脑测试：

### 在服务器上打包：

```bash
# 打包模型和配置文件
tar -czf k1_model_package.tar.gz \
    output/k1_add/k1_add_model.pt \
    data/envs/add_k1_env.yaml \
    data/agents/add_k1_agent.yaml \
    data/assets/k1.xml \
    data/motions/k1/ \
    args/add_k1_args.txt
```

### 下载到本地：

```bash
# 在本地电脑
scp your_server:~/workspace/MimicKit/k1_model_package.tar.gz .
```

### 在本地运行：

```bash
# 解压
tar -xzf k1_model_package.tar.gz

# 运行（带可视化）
python mimickit/run.py \
    --arg_file args/add_k1_args.txt \
    --mode test \
    --num_envs 1 \
    --visualize true \
    --model_file output/k1_add/k1_add_model.pt \
    --test_episodes 5
```

可以用OBS Studio或其他屏幕录制工具录制。

---

## 方法4: 修改代码保存帧图像（编程方式）

### 创建修改版的测试脚本 `test_and_save_frames.py`:

```python
import numpy as np
import os
import torch
from PIL import Image

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.arg_parser as arg_parser


def test_with_frame_capture(env, agent, output_dir, num_steps=300):
    """测试并尝试捕获帧"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    gym = env._engine._gym
    sim = env._engine._sim
    
    obs, info = env.reset()
    
    for step in range(num_steps):
        # 决策
        with torch.no_grad():
            action_info = agent.decide_action(obs, info)
            action = action_info["action"]
        
        # 执行
        obs, reward, done, info = env.step(action)
        
        # 尝试保存帧
        try:
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            # 这里需要camera sensor，见render_k1_video.py
        except:
            pass
        
        if step % 10 == 0:
            print(f"Step {step}/{num_steps}")


# 主函数
if __name__ == "__main__":
    import sys
    args = arg_parser.ArgParser()
    args.load_args(sys.argv[1:])
    
    # ... 加载环境和模型 ...
    # test_with_frame_capture(env, agent, "output/frames", 300)
```

---

## 方法5: 使用VNC查看（最直观）

### 在服务器上安装VNC：

```bash
sudo apt-get install tigervnc-standalone-server tigervnc-common
```

### 启动VNC server：

```bash
vncserver :1 -geometry 1280x720 -depth 24
```

### 在本地连接：

```bash
# 使用SSH隧道
ssh -L 5901:localhost:5901 your_server

# 然后用VNC viewer连接到 localhost:5901
```

### 在VNC会话中运行：

```bash
export DISPLAY=:1
CUDA_VISIBLE_DEVICES=6 python mimickit/run.py \
    --arg_file args/add_k1_args.txt \
    --mode test \
    --num_envs 1 \
    --visualize true \
    --model_file output/k1_add/k1_add_model.pt
```

然后用本地的屏幕录制工具录制VNC窗口。

---

## 快速对比

| 方法 | 难度 | 质量 | 需要X Server | 推荐度 |
|------|------|------|--------------|--------|
| Xvfb + ffmpeg | ⭐⭐ | ⭐⭐⭐⭐ | 虚拟 | ⭐⭐⭐⭐⭐ |
| EGL渲染 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 否 | ⭐⭐⭐ |
| 本地运行 | ⭐ | ⭐⭐⭐⭐⭐ | 是 | ⭐⭐⭐⭐ |
| 代码捕获帧 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 是 | ⭐⭐ |
| VNC远程 | ⭐⭐⭐ | ⭐⭐⭐ | 虚拟 | ⭐⭐⭐ |

---

## 推荐工作流程（最简单）

### 第一次尝试（Xvfb方法）：

```bash
# 1. 安装依赖
sudo apt-get install xvfb ffmpeg

# 2. 直接运行（简单测试）
CUDA_VISIBLE_DEVICES=6 xvfb-run -a python mimickit/run.py \
    --arg_file args/add_k1_args.txt \
    --mode test \
    --num_envs 1 \
    --visualize true \
    --model_file output/k1_add/k1_add_model.pt \
    --test_episodes 1
```

如果上面能运行，说明Xvfb工作正常，然后再添加录制功能。

---

## 故障排除

### 问题1: "Unable to access X display"

**解决方案**：使用xvfb-run或设置DISPLAY
```bash
export DISPLAY=:99
Xvfb :99 -screen 0 1280x720x24 &
```

### 问题2: Isaac Gym无法初始化

**解决方案**：检查CUDA和显卡驱动
```bash
nvidia-smi  # 确认GPU可用
echo $LD_LIBRARY_PATH  # 确认包含正确的库路径
```

### 问题3: ffmpeg录制失败

**解决方案**：手动测试ffmpeg
```bash
# 测试X11抓取
ffmpeg -f x11grab -video_size 1280x720 -i :99 -t 5 test.mp4
```

---

## 额外资源

- Isaac Gym文档: https://developer.nvidia.com/isaac-gym
- Xvfb手册: `man xvfb`
- FFmpeg文档: https://ffmpeg.org/documentation.html

---

**建议**: 首先尝试方法1（Xvfb），这是最平衡的方案。如果需要更高质量，考虑方法3（本地运行）。

