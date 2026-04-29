from pathlib import Path

import torch

from pygoo.config import RuntimeConfig, SimulationConfig
from pygoo.io.scene import load_scene
from pygoo.optimize import DifferentiableFitter
from pygoo.render import Viewport
from pygoo.trajectory import (
    compare_scene_json_trajectories,
    fitter_trajectory_distance,
    rollout_state_trajectory,
)


def test_compare_scene_json_trajectories_identical_is_zero(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "data" / "1" / "orig.json"
    if not src.exists():
        return

    copy_path = tmp_path / "copy.json"
    copy_path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    metrics = compare_scene_json_trajectories(
        str(src),
        str(copy_path),
        device="cpu",
        frames=8,
        substeps=2,
    )
    assert metrics.rmse == 0.0
    assert torch.all(metrics.per_frame_rmse == 0.0)


def test_trajectory_distance_gets_closer_after_opt_steps():
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "data" / "1" / "orig.json"
    if not src.exists():
        return

    cfg_target, state_target = load_scene(str(src), "cpu")
    state_train = state_target.clone()
    state_train.mass = state_train.mass * 1.7
    state_train.edge_k = state_train.edge_k * 0.65

    run_cfg = RuntimeConfig(width=1240, height=800, panel_width=420, substeps=4)
    fitter = DifferentiableFitter(
        run_cfg,
        num_levels=run_cfg.render_pyramid_levels,
        sigma_world=run_cfg.render_particle_sigma_world,
        kernel_size=run_cfg.render_pyramid_kernel_size,
    )
    fitter.configure(
        train_mass=True,
        train_edge_k=True,
        lr=0.003,
        min_advance_loss=0.0,
        temporal_discount=0.97,
        horizon_step=2,
        max_horizon=96,
    )

    vp = Viewport(
        run_cfg.world_x_min,
        run_cfg.world_x_max,
        run_cfg.world_y_min,
        run_cfg.world_y_max,
        run_cfg.panel_width,
        0,
        run_cfg.width - run_cfg.panel_width,
        run_cfg.height,
    )

    target_traj = rollout_state_trajectory(
        state_target,
        cfg_target,
        substeps=run_cfg.substeps,
        frames=10,
    )

    fitter._target_pyramids = []
    work = state_target.clone()
    for t in range(12):
        from pygoo.render import render_gaussian_pyramid

        fitter._target_pyramids.append(
            render_gaussian_pyramid(
                work,
                vp,
                vp.screen_h,
                vp.screen_w,
                num_levels=run_cfg.render_pyramid_levels,
                sigma_world=run_cfg.render_particle_sigma_world,
                kernel_size=run_cfg.render_pyramid_kernel_size,
                floor_y=cfg_target.floor_y,
                floor_enabled=cfg_target.floor_enabled,
            )
        )
        if t + 1 < 12:
            from pygoo.physics.step import Stepper

            Stepper(cfg_target, use_compile=False).rollout(work, run_cfg.substeps)
    fitter._target_device = "cpu"

    before = fitter_trajectory_distance(
        fitter,
        state_train,
        cfg_target,
        target_traj,
        substeps=run_cfg.substeps,
    ).rmse

    dists = []
    for _ in range(3):
        r = fitter.step(state_train, cfg_target, vp, run_cfg.substeps)
        assert r.ok
        d = fitter_trajectory_distance(
            fitter,
            state_train,
            cfg_target,
            target_traj,
            substeps=run_cfg.substeps,
        ).rmse
        dists.append(d)

    assert dists[0] <= before
    assert dists[1] <= dists[0]
    assert dists[2] <= dists[1]
