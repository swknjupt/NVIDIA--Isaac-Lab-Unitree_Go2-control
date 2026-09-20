# Unitree Go2 四任务强化学习工作流与训练计划

本文档用于指导在现有远程仿真环境（宿主机 `192.168.14.98` + `isaac-sim-kevin` 容器）下，系统化推进 Unitree Go2 四足机器人 4 个递进强化学习任务的训练与验收。

---

## 一、整体推进策略与核心原则

本项目聚焦于在虚拟仿真环境下实现四足机器人的高级运动与导航控制，采用**能力递进（Curriculum）**和**权重继承（Warm-start Transfer）**机制。

当前重点实施**阶段 1 至阶段 3** 的仿真主线闭环；**阶段 4（Sim2Real / RMA 抗扰）**作为后续真机部署的**可选扩展阶段**暂不作为当前训练的阻塞条件：

```text
[阶段 0: 环境准备与冒烟验证]
       │
       ▼
[阶段 1: Task 1 平地运动] ── (导出 go2_task1_model.pt) ──┐
       │                                                 │ (Warm-start)
       ▼ (Warm-start)                                    ▼
[阶段 2: Task 2 多地形盲爬] ───────────────┬──────> [阶段 3: Task 3 导航与雷达避障]
       │ (导出 go2_task2_model.pt)          │        (导出 go2_task3_model.pt)
       │                                    │        ★ 核心主线达成 ★
       ┆ (可选扩展 / 真机部署前置)           ┆
       ▼                                    ▼
[可选: Task 4 Sim2Real / RMA 抗扰] <───────┘
       (导出 go2_task4_teacher_model.pt)
```

### 核心实施原则：
1. **严禁“直接盲跑全量”**：每个 Task 正式训练前，必须按顺序跑通 **World 逻辑测试 $\rightarrow$ Env 接口测试 $\rightarrow$ Smoke 冒烟测试**，验证数值稳定性（防 NaN/Inf）与日志写入。
2. **渐进增加并发环境数**：先以 `num_envs=16` 验证 API 与显存，再以 `num_envs=512 ~ 1024` 进行正式大规模并行训练。
3. **分阶段验收与 Checkpoint 留档**：每个阶段需达到明确的定量指标后，才将 Checkpoint 作为下一个阶段的 Warm-start 底座。

---

## 二、运行环境准备与配置说明

当前远程服务器 `192.168.14.98` 具备 2 × RTX A5000（每张 24GB 显存）。仿真环境运行在容器 `isaac-sim-kevin` 中。

### 1. 容器内执行规范
容器内推荐使用 Isaac Sim 官方 Python 入口启动训练与测试脚本：
```bash
/isaac-sim/python.sh <脚本路径> [参数]
# 或者
/workspace/isaaclab/isaaclab.sh -p <脚本路径> [参数]
```

### 2. 代码同步与路径映射
由于代码保存在宿主机 `/home/unitree_go2_isaaclab_rl`，在容器内运行时建议挂载或同步到 `/workspace/go2_rl`：
- **容器内代码根目录**：`/workspace/go2_rl`
- **容器内日志根目录**：`/workspace/go2_rl/logs`
- **Isaac Lab 根目录**：`/workspace/isaaclab`

配置私有路径文件 `configs/local_paths.yaml`：
```yaml
ubuntu:
  project_root: "/workspace/go2_rl"
  isaaclab_root: "/workspace/isaaclab"
  log_root: "/workspace/go2_rl/logs"
```

---

## 三、分阶段详细执行计划与验收标准

### 阶段 0：前置环境与集成测试（Day 1 上午）

- **目标**：验证容器环境、Isaac Sim 渲染/物理引擎、Go2 资产、Gymnasium API。
- **执行内容**：
  1. 静态检查：`python3 -m py_compile` 确认无语法错误；
  2. 运行 `tests/task2/task2_world_test.py`（白盒验证地形生成器与高度采样）；
  3. 运行 `tests/task3/task3_world_test.py`（白盒验证 60 线雷达与动静态障碍物解析几何）；
  4. 依次执行 4 个任务的 Env 测试（`tests/task*/task*_env_test.py`，`--num-envs 8 --steps 50`）。
- **验收标准**：
  - [ ] 所有测试脚本输出 `[OK]`，无 NaN/Inf 报错；
  - [ ] 显存正常分配（单机占用 < 4GB），无 Vulkan/CUDA 崩溃；
  - [ ] Task2 地形测试出现明确标记：`Go2 Task2 World / Terrain / Curriculum 测试全部通过`。

---

### 阶段 1：Task 1 平地速度跟踪（Day 1 ~ Day 2）

- **任务目标**：训练 Go2 具备稳健的平地步态，精确跟踪线速度 $v_x, v_y$ 和角速度 $\omega_z$。
- **状态空间**：Actor Obs 87 维，Privileged Obs 0 维（对称 Actor-Critic），Action 12 维。
- **前置依赖**：无（从零开始训练，Random Weights）。
- **执行步骤**：
  1. **Smoke 测试**：
     ```bash
     /isaac-sim/python.sh src/go2_rl/tasks/task1/task1_train.py \
         --num-envs 64 --total-env-steps 65536 --headless
     ```
  2. **全量正式训练**：
     ```bash
     /isaac-sim/python.sh src/go2_rl/tasks/task1/task1_train.py \
         --num-envs 512 --total-env-steps 350000000 --rollouts 64 \
         --learning-epochs 5 --mini-batches 8 --headless --device cuda:0
     ```
  3. **评估与自动录像导出**：
     ```bash
     # 运行无头评测并自动录制导出 1080P/720P MP4 视频
     /isaac-sim/python.sh src/go2_rl/tasks/task1/task1_model_test.py \
         --checkpoint logs/task1/<run_name>/final_checkpoint/go2_task1_model.pt \
         --num-envs 16 --steps 2000 --headless-eval --record-video --video-length 600
     # 或者使用 shell 脚本快捷入口:
     # bash scripts/ubuntu/eval_task1.sh logs/task1/<run_name>/final_checkpoint/go2_task1_model.pt --record-video
     ```
- **验收标准**：
  - **量化指标验收**：
    - [ ] 摔倒率（Termination Rate）：最后 5000 万步内摔倒率 $< 1.0\%$；
    - [ ] 速度跟踪误差：线速度误差 $\le 0.15\ \text{m/s}$，角速度误差 $\le 0.2\ \text{rad/s}$；
    - [ ] 躯干高度：维持在 $0.28 \sim 0.33\ \text{m}$，无剧烈俯仰与翻滚晃动；
    - [ ] Checkpoint 产物：`logs/task1/<run_name>/final_checkpoint/go2_task1_model.pt` 成功导出。
  - **视觉步态与视频验收支撑**：
    - [ ] 视频交付物：自动生成并导出 `logs/task1/<run_name>/eval_videos/task1_eval.mp4`（约 20 秒 30FPS 高清录像）；
    - [ ] 步态观感：呈现对称稳定的对角小跑步态（Trot），足端离地清晰无拖行拖拽，无高频关节抽搐（Jittering）。

---

### 阶段 2：Task 2 多地形与盲爬（Day 2 ~ Day 4）

- **任务目标**：引入粗糙平地、斜坡、梅花桩和阶梯 4 种地形（10 级难度递进），基于 91 维高度图扫描和 Asymmetric Critic 训练适应性步态。
- **状态空间**：Actor Obs 87 维，Privileged Obs 178 维（87 Actor + 91 地形特权特征），Action 12 维。
- **前置依赖**：Warm-start 继承 Task 1 导出的 Checkpoint。
- **执行步骤**：
  1. **Smoke 测试**：加载 Task 1 模型，跑 64 环境快速验证 Asymmetric Critic 维度。
  2. **全量正式训练**：
     ```bash
     /isaac-sim/python.sh src/go2_rl/tasks/task2/task2_train.py \
         --pretrained-task1 logs/task1/<run_name>/final_checkpoint/go2_task1_model.pt \
         --num-envs 512 --total-env-steps 350000000 --headless --device cuda:0
     ```
  3. **评估与自动录像导出**：
     ```bash
     /isaac-sim/python.sh src/go2_rl/tasks/task2/task2_model_test.py \
         --checkpoint logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt \
         --num-envs 16 --steps 3000 --headless-eval --record-video --video-length 600
     # 或者使用 shell 脚本快捷入口:
     # bash scripts/ubuntu/eval_task2.sh logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt --record-video
     ```
- **验收标准**：
  - **量化指标验收**：
    - [ ] 地形课程晋升率：至少 $60\%$ 的环境达到 Level 7 以上难度，能够稳定通过台阶和梅花桩；
    - [ ] 粗糙地面与斜坡通过率：坡度 15° 范围内上坡与下坡不打滑、不倾覆；
    - [ ] 足端卡住/碰撞惩罚：足端撞击阶梯立面的惩罚平稳收敛；
    - [ ] Checkpoint 产物：`logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt` 成功导出。
  - **视觉步态与视频验收支撑**：
    - [ ] 视频交付物：自动生成并导出 `logs/task2/<run_name>/eval_videos/task2_eval.mp4`；
    - [ ] 越障观感：台阶攀爬时足端主动抬高跨越立面，无卡死绊倒，坡道行进中身体姿态自主俯仰平衡。

---

### 阶段 3：Task 3 目标导航与激光雷达避障（Day 4 ~ Day 6）

- **任务目标**：结合 60 束激光雷达观测、目标航向导引，在包含 25 个静态圆柱障碍物与 8 个动态移动障碍物的复杂环境中自主导航。
- **状态空间**：Actor Obs 208 维（含 60 维 Lidar、目标相对向量与运动状态），Privileged Obs 276 维，Action 12 维。
- **前置依赖**：Warm-start 继承 Task 1 或 Task 2 的底座步态。
- **执行步骤**：
  1. **Smoke 测试**：
     ```bash
     /isaac-sim/python.sh src/go2_rl/tasks/task3/task3_train.py \
         --num-envs 64 --total-env-steps 65536 --headless
     ```
  2. **全量正式训练**（分 6 阶段课程）：
     ```bash
     /isaac-sim/python.sh src/go2_rl/tasks/task3/task3_train.py \
         --pretrained-locomotion logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt \
         --num-envs 512 --total-env-steps 900000000 --headless --device cuda:0
     ```
  3. **导航成功率测试与自动录像导出**：
     ```bash
     /isaac-sim/python.sh src/go2_rl/tasks/task3/task3_model_test.py \
         --checkpoint logs/task3/<run_name>/final_checkpoint/go2_task3_model.pt \
         --num-envs 16 --steps 3000 --headless-eval --record-video --video-length 600
     # 或者使用 shell 脚本快捷入口:
     # bash scripts/ubuntu/eval_task3.sh logs/task3/<run_name>/final_checkpoint/go2_task3_model.pt --record-video
     ```
- **验收标准**：
  - **量化指标验收**：
    - [ ] 目标到达成功率（Success Rate）：在障碍物密度正常场景下，目标点到达率 $\ge 85\%$；
    - [ ] 碰撞率（Collision Rate）：动静态障碍物碰撞率 $< 8\%$；
    - [ ] 避障平滑度：接近障碍物时能提前减速并绕行，无剧烈抽搐抖动；
    - [ ] Checkpoint 产物：`logs/task3/<run_name>/final_checkpoint/go2_task3_model.pt` 成功导出。
  - **视觉避障与视频验收支撑**：
    - [ ] 视频交付物：自动生成并导出 `logs/task3/<run_name>/eval_videos/task3_eval.mp4`；
    - [ ] 航向与绕行观感：画面中跟随视角可清晰看到目标点球体（Goal）与圆柱障碍物，机器狗在距离障碍物约 1~1.5m 处提前变向平滑绕行，平稳抵达目标区域。

---

### [可选 / 扩展阶段] 阶段 4：Task 4 Sim2Real / RMA 鲁棒性抗扰训练（未来有真机部署需求时启动）

- **任务定位**：本项目前 3 个阶段已完成在仿真环境内的平地、复杂地形与雷达避障闭环。本阶段专门用于弥合真实物理环境与仿真的 Reality Gap（域随机化与抗扰），当前仿真主线阶段暂不执行，作为未来面向物理真机部署时的预留扩展。
- **任务目标**：通过大规模动力学域随机化（摩擦系数 0.2~1.25、附加质量 0~5kg、质心偏移、电机强度衰减、突发外力推扰），训练高鲁棒性的 RMA Teacher Policy。
- **状态空间**：Single Actor Obs 48 维，5 帧历史堆叠为 240 维，特权物理量 25 维，Teacher 观测共 265 维，Action 12 维。
- **前置依赖**：Warm-start 继承 Task 2 的多地形权重。
- **执行步骤**：
  1. **Smoke 测试**：验证 Frame Stack Wrapper 历史维度（240维）与特权物理量（25维）拼接正确。
  2. **全量正式训练**：
     ```bash
     /isaac-sim/python.sh src/go2_rl/tasks/task4/task4_train.py \
         --pretrained-task2 logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt \
         --num-envs 512 --total-env-steps 400000000 --headless --device cuda:0
     ```
  3. **极端抗扰鲁棒性评估**：
     ```bash
     /isaac-sim/python.sh src/go2_rl/tasks/task4/task4_model_test.py \
         --checkpoint logs/task4/<run_name>/final_checkpoint/go2_task4_teacher_model.pt \
         --num-envs 16 --steps 3000 --headless-eval
     ```
- **量化验收标准**：
  - [ ] 外力推扰恢复能力：遭受瞬间横向冲量（$1.5\sim 2.5\ \text{m/s}$ 等效推力）后能在 1.5 秒内自动调整步频恢复平衡；
  - [ ] 负载适应性：在背负 3~5kg 偏心载荷及低摩擦地面（$\mu = 0.3$）下依然能稳定行进；
  - [ ] 摔倒率：综合域随机化全开条件下，摔倒率 $< 3.0\%$；
  - [ ] Checkpoint 产物：`logs/task4/<run_name>/final_checkpoint/go2_task4_teacher_model.pt` 成功导出。

---

## 四、训练资源监控与排错指南

1. **显存监控与并发控制**：
   - 训练期间在宿主机保持监控：
     ```bash
     watch -n 2 nvidia-smi
     ```
   - 若出现 CUDA Out-of-Memory，通过命令行降低 `--num-envs`（如从 512 降至 256 或 128），并调小 `--rollouts`。
2. **TensorBoard 实时查看**：
   - 启动 TensorBoard 服务映射：
     ```bash
     tensorboard --logdir logs/ --port 6006 --bind_all
     ```
   - 重点监控曲线：`Reward/Total`（总奖励）、`Episode/Length`（存活步数）、`Loss/Policy`、`Loss/Value`、`Curriculum/Level`。
3. **Isaac Kit 退出生命周期注意**：
   - 在测试结束退出时，Kit 可能会花费数十秒清理 Vulkan 上下文，只要日志中打出了 `[OK]` 及测试通过摘要，即可视为该项验证成功。
