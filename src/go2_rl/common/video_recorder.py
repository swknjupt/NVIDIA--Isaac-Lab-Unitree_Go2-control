# Copyright (c) 2026
# Unitree Go2 Common: 自动录屏与 MP4 视频导出工具。
#
# 本文件提供面向 Isaac Lab 评测环境的轻量级视频录制器。
# 主要职责:
#   1. 通过 omni.replicator.core 创建 RGB 捕获器，支持 headless 模式下的离线高清渲染；
#   2. 支持第三人称平滑视角跟踪 (Following Camera)，动态聚焦环境 0 的机器狗；
#   3. 通过 imageio 将帧序列自动编码为通用 H.264 MP4 视频并保存到指定路径；
#   4. 提供优雅的异常回退与资源释放机制，保证录像过程不影响核心评测与指标统计。
#
# 使用方式:
#   recorder = Go2VideoRecorder(env=base_env, video_path="logs/.../eval_videos/task1_eval.mp4", fps=30)
#   recorder.step(robot_pos=root_pos_0)
#   recorder.close()

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np


class Go2VideoRecorder:
    """Video recorder for Unitree Go2 Isaac Lab evaluation rollouts."""

    def __init__(
        self,
        env: Any,
        video_path: str | Path,
        fps: int = 30,
        resolution: Tuple[int, int] = (1280, 720),
        record_steps: int = 500,
        camera_prim_path: str = "/OmniverseKit_Persp",
        follow_robot: bool = True,
        camera_offset: Sequence[float] = (-2.8, -2.8, 1.8),
    ):
        self.env = env
        self.video_path = Path(video_path).expanduser().resolve()
        self.fps = int(fps)
        self.resolution = tuple(resolution)
        self.record_steps = int(record_steps)
        self.camera_prim_path = str(camera_prim_path)
        self.follow_robot = bool(follow_robot)
        self.camera_offset = np.array(camera_offset, dtype=np.float32)

        self.video_path.parent.mkdir(parents=True, exist_ok=True)

        self._render_product = None
        self._annotator = None
        self._writer = None
        self.frame_count = 0
        self.is_active = False
        self._smoothed_target: Optional[np.ndarray] = None

        self._init_recorder()

    def _init_recorder(self) -> None:
        try:
            import imageio

            try:
                self._writer = imageio.get_writer(
                    str(self.video_path),
                    fps=self.fps,
                    codec="libx264",
                    quality=8,
                    pixelformat="yuv420p",
                )
            except Exception:
                # Fallback to default codec if libx264 options fail
                self._writer = imageio.get_writer(str(self.video_path), fps=self.fps)

            import omni.replicator.core as rep

            self._render_product = rep.create.render_product(
                self.camera_prim_path, self.resolution
            )
            self._annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._annotator.attach([self._render_product])

            self.is_active = True
            print(f"[INFO] Go2VideoRecorder initialized. Recording to: {self.video_path}")
            print(f"[INFO] Video resolution: {self.resolution[0]}x{self.resolution[1]}, FPS: {self.fps}")
        except Exception as exc:
            self.is_active = False
            print(f"[WARN] Failed to initialize Go2VideoRecorder ({exc}). Video recording disabled.")

    def step(self, robot_pos: Optional[Sequence[float]] = None) -> None:
        """Capture one video frame, optionally updating camera tracking."""
        if not self.is_active or self._writer is None:
            return

        if self.frame_count >= self.record_steps:
            return

        try:
            # 1. Update following camera if requested
            if self.follow_robot and robot_pos is not None:
                cur_pos = np.array(robot_pos[:3], dtype=np.float32)
                if self._smoothed_target is None:
                    self._smoothed_target = cur_pos
                else:
                    self._smoothed_target = 0.85 * self._smoothed_target + 0.15 * cur_pos

                target = self._smoothed_target.copy()
                # Focus slightly above the base height
                target[2] = max(target[2], 0.25)
                eye = target + self.camera_offset

                if hasattr(self.env, "sim") and hasattr(self.env.sim, "set_camera_view"):
                    self.env.sim.set_camera_view(
                        eye=eye.tolist(),
                        target=target.tolist(),
                        camera_prim_path=self.camera_prim_path,
                    )

            # 2. Render simulation frame
            if hasattr(self.env, "sim") and hasattr(self.env.sim, "render"):
                self.env.sim.render()

            # 3. Fetch annotated RGB frame
            if self._annotator is not None:
                data = self._annotator.get_data()
                if data is not None and getattr(data, "size", 0) > 0:
                    arr = np.frombuffer(data, dtype=np.uint8).reshape(*data.shape)
                    rgb = arr[:, :, :3]
                    self._writer.append_data(rgb)
                    self.frame_count += 1
        except Exception as exc:
            print(f"[WARN] Frame capture failed at step {self.frame_count}: {exc}")

    def close(self) -> None:
        """Finalize video encoding and report status."""
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception as exc:
                print(f"[WARN] Error closing video writer: {exc}")
            self._writer = None

        if self.frame_count > 0 and self.video_path.exists():
            file_size_mb = self.video_path.stat().st_size / (1024 * 1024)
            duration = self.frame_count / max(self.fps, 1)
            print("\n" + "=" * 80)
            print("[OK] Video recording finished successfully!")
            print(f"  Output MP4 : {self.video_path}")
            print(f"  Frames     : {self.frame_count} frames")
            print(f"  Duration   : {duration:.2f} seconds ({self.fps} FPS)")
            print(f"  File size  : {file_size_mb:.2f} MB")
            print("=" * 80 + "\n")
        elif self.is_active:
            print(f"[WARN] No frames recorded to {self.video_path}")
        self.is_active = False
