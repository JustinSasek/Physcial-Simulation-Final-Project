from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import SimulationConfig
from .io.scene import load_scene_or_observable
from .optimize import DifferentiableFitter
from .physics import Stepper
from .state import SimulationState


@dataclass
class TrajectoryMetrics:
    rmse: float
    per_frame_rmse: torch.Tensor


def trajectory_rmse_series(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError("Trajectory tensors must have same shape")
    diff = pred - target
    return torch.sqrt(torch.mean(diff.pow(2), dim=(1, 2)))


def rollout_state_trajectory(
    state: SimulationState,
    sim_cfg: SimulationConfig,
    substeps: int,
    frames: int,
) -> torch.Tensor:
    if frames <= 0:
        return torch.empty((0, 0, 2), dtype=state.pos.dtype, device=state.device)

    stepper = Stepper(sim_cfg, use_compile=False)
    work = state.clone()
    traj: list[torch.Tensor] = []
    for t in range(frames):
        traj.append(work.pos)
        if t + 1 < frames:
            stepper.rollout(work, substeps)
    return torch.stack(traj, dim=0)


def trajectory_metrics(pred: torch.Tensor, target: torch.Tensor) -> TrajectoryMetrics:
    per_frame = trajectory_rmse_series(pred, target)
    return TrajectoryMetrics(
        rmse=float(torch.mean(per_frame).item()),
        per_frame_rmse=per_frame,
    )


def save_trajectory_npz(
    path: str,
    traj: torch.Tensor,
    *,
    substeps: int,
    metadata: dict[str, float] | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "pos": traj.detach().cpu().numpy().astype(np.float32),
        "substeps": np.array([int(substeps)], dtype=np.int32),
    }
    if metadata:
        for k, v in metadata.items():
            payload[k] = np.array([float(v)], dtype=np.float32)
    np.savez_compressed(str(p), **payload)  # type: ignore[arg-type]


def load_trajectory_npz(
    path: str, device: str = "cpu"
) -> tuple[torch.Tensor, dict[str, float]]:
    with np.load(path) as data:
        if "pos" not in data:
            raise ValueError("Trajectory NPZ missing 'pos'")
        pos = torch.tensor(
            data["pos"], dtype=torch.float32, device=torch.device(device)
        )
        meta: dict[str, float] = {}
        for key in data.files:
            if key == "pos":
                continue
            arr = data[key]
            if np.isscalar(arr) or (hasattr(arr, "size") and arr.size == 1):
                meta[key] = float(np.asarray(arr).reshape(-1)[0])
    return pos, meta


def compare_scene_json_trajectories(
    path_a: str,
    path_b: str,
    device: str,
    frames: int,
    substeps: int,
    default_spring_stiffness: float = 100.0,
) -> TrajectoryMetrics:
    cfg_a, state_a = load_scene_or_observable(path_a, device, default_spring_stiffness)
    cfg_b, state_b = load_scene_or_observable(path_b, device, default_spring_stiffness)
    traj_a = rollout_state_trajectory(state_a, cfg_a, substeps, frames)
    traj_b = rollout_state_trajectory(state_b, cfg_b, substeps, frames)
    return trajectory_metrics(traj_a, traj_b)


def fitter_trajectory_distance(
    fitter: DifferentiableFitter,
    state: SimulationState,
    sim_cfg: SimulationConfig,
    target_traj: torch.Tensor,
    substeps: int,
) -> TrajectoryMetrics:
    pred = fitter.rollout_trajectory(
        state,
        sim_cfg,
        substeps=substeps,
        frames=int(target_traj.shape[0]),
    )
    return trajectory_metrics(pred, target_traj)
