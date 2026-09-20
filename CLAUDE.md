# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository is a modular reinforcement learning framework for the 12-DoF Unitree Go2 quadruped robot built on NVIDIA Isaac Lab (Isaac Sim) and skrl (PPO). It comprises 4 progressive tasks:
1. **Task 1 (`task1_flat_locomotion`)**: Flat-ground velocity tracking (`vx`, `vy`, `wz`).
2. **Task 2 (`task2_multiterrain`)**: Multi-terrain locomotion with height scanning and terrain curriculum.
3. **Task 3 (`task3_navigation`)**: Autonomous navigation, obstacle avoidance (static & dynamic), and 60-ray lidar sensing.
4. **Task 4 (`task4_sim2real_rma`)**: Sim2Real / RMA robust locomotion teacher policy with domain randomization.

---

## Environment & Setup

- **Runtime Environment**: Python 3.10+ (typically Python 3.11 inside the Isaac Lab conda environment).
- **Core Dependencies**: Isaac Lab, Isaac Sim, PyTorch (CUDA enabled), skrl, TensorBoard, tqdm, numpy.
- **Python Path**: Direct execution of Python modules requires `export PYTHONPATH=$PWD/src:$PYTHONPATH`. The shell and PowerShell scripts under `scripts/` configure this automatically.
- **Path Configuration**: Log locations and external tool paths can be specified via `configs/local_paths.yaml` (copy from `configs/local_paths.example.yaml`) or environment variables `RT_GO2_LOG_ROOT` and `RT_GO2_TASK<N>_LOG_ROOT`.

---

## Common Development Commands

### 1. Static Verification
Run static checks before launching the simulator:
```bash
# Compile all python files
python3 -m py_compile $(find src tests -name "*.py")

# Check Ubuntu shell script syntax
bash -n scripts/ubuntu/*.sh
```

### 2. Environment Verification
```bash
bash scripts/ubuntu/check_env.sh
# Windows PowerShell: .\scripts\windows\check_env.ps1
```

### 3. World & Environment Tests
Run test suites via Ubuntu scripts:
```bash
# World-model white-box tests (terrain / navigation logic)
bash scripts/ubuntu/test_task2_world.sh [--num-envs 1000]
bash scripts/ubuntu/test_task3_world.sh [--num-envs 512]

# IsaacLab environment tests (Gymnasium reset/step API, rollout stability)
bash scripts/ubuntu/test_task1_env.sh [--num-envs 64] [--steps 300]
bash scripts/ubuntu/test_task2_env.sh [--num-envs 8] [--steps 16]
bash scripts/ubuntu/test_task3_env.sh [--num-envs 8] [--steps 16]
bash scripts/ubuntu/test_task4_env.sh [--num-envs 8] [--steps 16]
```

To run a single test file directly with Python:
```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
python tests/task1/task1_env_test.py --num-envs 8 --steps 50 --headless --device cuda:0
python tests/task2/task2_world_test.py --num-envs 64 --headless --test-device cuda:0
```

### 4. Smoke Training (Minimal Pipeline Test)
Verify the PPO rollout, logging, and checkpoint pipeline without long training runs:
```bash
bash scripts/ubuntu/smoke_task1.sh
bash scripts/ubuntu/smoke_task2.sh
bash scripts/ubuntu/smoke_task3.sh
bash scripts/ubuntu/smoke_task4.sh
```

### 5. Formal Training
```bash
# Task 1 (from scratch)
bash scripts/ubuntu/train_task1.sh [--num-envs 512] [--total-env-steps 350000000]

# Task 2 (supports warm-start from Task 1 checkpoint)
bash scripts/ubuntu/train_task2.sh [logs/task1/<run_name>/final_checkpoint/go2_task1_model.pt]

# Task 3 (supports warm-start from Task 2 checkpoint)
bash scripts/ubuntu/train_task3.sh [logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt]

# Task 4 (supports warm-start from Task 2 checkpoint)
bash scripts/ubuntu/train_task4.sh [logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt]
```

### 6. Evaluation and Visualization
```bash
# Headless model evaluation & metrics table
bash scripts/ubuntu/eval_task1.sh <path_to_checkpoint.pt>
bash scripts/ubuntu/eval_task2.sh <path_to_checkpoint.pt>
bash scripts/ubuntu/eval_task3.sh <path_to_checkpoint.pt>
bash scripts/ubuntu/eval_task4.sh <path_to_checkpoint.pt>

# Headless evaluation with automatic MP4 video recording
bash scripts/ubuntu/eval_task1.sh <path_to_checkpoint.pt> --record-video [--video-length 600]
bash scripts/ubuntu/eval_task2.sh <path_to_checkpoint.pt> --record-video [--video-length 600]
bash scripts/ubuntu/eval_task3.sh <path_to_checkpoint.pt> --record-video [--video-length 600]

# GUI visualization (requires display)
bash scripts/ubuntu/visualize_task1.sh <path_to_checkpoint.pt>
bash scripts/ubuntu/visualize_task2.sh <path_to_checkpoint.pt>
bash scripts/ubuntu/visualize_task3.sh <path_to_checkpoint.pt>
bash scripts/ubuntu/visualize_task4.sh <path_to_checkpoint.pt>
```

---

## High-Level Architecture & Code Structure

### Task Hierarchy & Observation Dimensions

| Task | Actor Obs Dim | Privileged Obs Dim | Action Dim | Description |
|---|---|---|---|---|
| **Task 1** | 87 | 0 (symmetric) | 12 | Flat velocity tracking (`vx`, `vy`, `wz`), joint positions/velocities, gravity projection, last action. |
| **Task 2** | 87 | 178 (87 actor + 91 terrain privileged tail) | 12 | 4 terrains (rough flat, slopes, stepping stones, stairs), 10 curriculum levels, height scan grid. |
| **Task 3** | 208 | 276 (208 actor + 68 world privileged tail) | 12 | 60 lidar rays, target position, static & dynamic cylindrical obstacles, risk alerts. |
| **Task 4** | 240 (5 frames × 48) | 25 (265 teacher obs total) | 12 | Sim2Real teacher-student setup. Teacher uses 265 obs. Domain randomization for friction, payload, COM, motor degradation, push. |

### Module Organization

- **`src/go2_rl/common/`**: Shared infrastructure across all tasks:
  - `go2_skrl_models.py`: `Go2Actor` (Gaussian policy) and `Go2Critic` (deterministic value network) supporting asymmetric state spaces.
  - `go2_skrl_wrappers.py`: Gymnasium/IsaacLab environment wrappers implementing frame-stacking, asymmetric observation splitting (`states` vs `observations`), and info metric extraction.
  - `checkpoint_utils.py` & `normalizer_utils.py`: Policy checkpoint loading/saving and empirical observation normalizer state management.
  - `paths.py`: Log root resolution with fallback precedence: explicit arg -> task-specific env var -> general env var -> `local_paths.yaml` -> `<project_root>/logs/<task>`.
  - `progress.py` & `train_metadata.py`: Training progress display and run metadata persistence.
  - `video_recorder.py`: Autonomous headless RGB frame capture with camera tracking and H.264 MP4 export (`Go2VideoRecorder`).

- **`src/go2_rl/tasks/task{1..4}/`**: Self-contained per-task implementation:
  - `task*_config.py`: Pure dataclass configuration (no IsaacLab / Omniverse imports, safe to load anywhere).
  - `task*_world.py` (Tasks 2 & 3): Analytical terrain and navigation logic decoupled from IsaacLab, enabling fast GPU/torch unit testing.
  - `task*_env.py`: Gymnasium-compliant RL environment wrapping IsaacLab actors, sensors, and simulation dynamics.
  - `task*_train.py`: skrl PPO training entry point.
  - `task*_model_test.py`: Standalone evaluation and visualization runner.

- **`tests/task{1..4}/`**: Integration and unit tests:
  - `task*_world_test.py`: White-box validation of procedural terrain generation, coordinate origin mappings, obstacle dynamics, and lidar raycasts.
  - `task*_env_test.py`: Gymnasium API validation (`reset()`, `step()`), tensor finite checks (NaN/Inf protection), and curriculum rollout checks.

- **`scripts/ubuntu/` & `scripts/windows/`**: Parameterized shell scripts for headless CI, training, testing, evaluation, and GUI visualization.

### Critical Engineering Notes

- **AppLauncher Startup Order**: Files importing `isaaclab` or `pxr` must initialize `isaaclab.app.AppLauncher` *before* importing any Isaac Sim / Omniverse modules. Config dataclasses (`task*_config.py`) and path utilities (`common/paths.py`) are strictly decoupled from IsaacLab so they can be imported without launching an Omniverse instance.
- **Isaac Sim Process Teardown**: In automated test scripts (such as `test_task2_world.sh`), the Omniverse Kit process shutdown may take additional time or require timeout handling; functional test success is signaled by the explicit stdout marker `[OK]` and `测试全部通过`.
