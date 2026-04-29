from pathlib import Path

import torch

from pygoo.config import RuntimeConfig, SimulationConfig
from pygoo.io.scene import load_observable_state
from pygoo.optimize import DifferentiableFitter
from pygoo.physics.step import Stepper
from pygoo.render import Viewport, render_gaussian_pyramid
from pygoo.state import SimulationState


def _make_state(cfg: SimulationConfig) -> SimulationState:
    state = SimulationState.empty("cpu")
    state.add_particle((-0.25, 0.25), cfg, fixed=False)
    state.add_particle((0.25, 0.15), cfg, fixed=False)
    state.add_particle((0.0, 0.35), cfg, fixed=False)
    if state.rest_len.numel() > 0:
        state.rest_len = state.rest_len * 0.75
    return state


def _make_target_pyramids(
    state: SimulationState,
    cfg: SimulationConfig,
    vp: Viewport,
    substeps: int,
    frames: int,
    num_levels: int,
) -> list[list[torch.Tensor]]:
    out: list[list[torch.Tensor]] = []
    stepper = Stepper(cfg, use_compile=False)
    sim = state.clone()
    for t in range(frames):
        out.append(
            render_gaussian_pyramid(sim, vp, vp.screen_h, vp.screen_w, num_levels)
        )
        if t + 1 < frames:
            stepper.rollout(sim, substeps)
    return out


def _build_optimizer(fitter: DifferentiableFitter, state: SimulationState) -> None:
    ok, msg = fitter.build_optimizer(state)
    assert ok, msg


def test_curriculum_advances_with_min_loss_gate():
    cfg = SimulationConfig(time_step=0.004, floor_enabled=False)
    run_cfg = RuntimeConfig(width=320, height=240, panel_width=80, substeps=3)
    vp = Viewport(
        -1.0, 1.0, -1.0, 1.0, 0, 0, run_cfg.width - run_cfg.panel_width, run_cfg.height
    )

    state = _make_state(cfg)
    fitter = DifferentiableFitter(
        run_cfg, num_levels=3, sigma_world=0.05, kernel_size=5
    )
    fitter.configure(
        train_mass=True,
        train_edge_k=True,
        lr=0.01,
        min_advance_loss=1.0,
        temporal_discount=0.97,
        horizon_step=2,
        max_horizon=18,
    )
    fitter.reset_curriculum()

    fitter._target_pyramids = _make_target_pyramids(
        state, cfg, vp, substeps=run_cfg.substeps, frames=8, num_levels=3
    )
    fitter._target_device = str(state.device)

    _build_optimizer(fitter, state)

    r1 = fitter.step(state, cfg, vp, run_cfg.substeps)
    assert r1.ok
    assert fitter.horizon == 6
    assert fitter.spatial_stage == 1

    r2 = fitter.step(state, cfg, vp, run_cfg.substeps)
    assert r2.ok
    assert fitter.horizon == 6
    assert fitter.spatial_stage == 2

    r3 = fitter.step(state, cfg, vp, run_cfg.substeps)
    assert r3.ok
    assert fitter.horizon == 6
    assert fitter.spatial_stage == 3


def test_curriculum_auto_advances_when_loss_stalls():
    cfg = SimulationConfig(time_step=0.004, floor_enabled=False)
    run_cfg = RuntimeConfig(width=320, height=240, panel_width=80, substeps=3)
    vp = Viewport(
        -1.0, 1.0, -1.0, 1.0, 0, 0, run_cfg.width - run_cfg.panel_width, run_cfg.height
    )

    state = _make_state(cfg)
    fitter = DifferentiableFitter(
        run_cfg, num_levels=3, sigma_world=0.05, kernel_size=5
    )
    fitter.configure(
        train_mass=True,
        train_edge_k=True,
        lr=0.01,
        loss_threshold=0.0,
        temporal_discount=0.97,
        horizon_step=2,
        max_horizon=18,
        auto_advance_no_improve=True,
        no_improve_patience=2,
        loss_improvement_epsilon=1.0,
    )
    fitter.reset_curriculum()

    target_state = state.clone()
    target_state.mass = target_state.mass * 1.4
    target_state.edge_k = target_state.edge_k * 0.7
    target_state.pos = target_state.pos + torch.tensor(
        [0.06, -0.04], dtype=target_state.pos.dtype
    )
    fitter._target_pyramids = _make_target_pyramids(
        target_state, cfg, vp, substeps=run_cfg.substeps, frames=8, num_levels=3
    )
    fitter._target_device = str(state.device)

    _build_optimizer(fitter, state)

    r1 = fitter.step(state, cfg, vp, run_cfg.substeps)
    assert r1.ok
    assert not r1.schedule_advanced
    assert r1.schedule_advance_reason == "none"

    r2 = fitter.step(state, cfg, vp, run_cfg.substeps)
    assert r2.ok
    assert not r2.schedule_advanced
    assert r2.schedule_advance_reason == "none"

    r3 = fitter.step(state, cfg, vp, run_cfg.substeps)
    assert r3.ok
    assert r3.schedule_advanced
    assert r3.schedule_advance_reason == "no-improve"
    assert fitter.horizon == 6


def test_step_fails_when_no_trainable_parameters_selected():
    cfg = SimulationConfig(time_step=0.004, floor_enabled=False)
    run_cfg = RuntimeConfig(width=220, height=180, panel_width=20, substeps=2)
    vp = Viewport(
        -1.0, 1.0, -1.0, 1.0, 0, 0, run_cfg.width - run_cfg.panel_width, run_cfg.height
    )

    state = _make_state(cfg)
    fitter = DifferentiableFitter(
        run_cfg, num_levels=2, sigma_world=0.05, kernel_size=5
    )
    fitter.configure(
        train_mass=False,
        train_edge_k=False,
        lr=0.01,
        min_advance_loss=0.1,
        temporal_discount=0.9,
        horizon_step=2,
        max_horizon=16,
        optimize_cfg={
            "time_step": False,
            "gravity_g": False,
            "particle_mass": False,
            "spring_stiffness": False,
            "damping_stiffness": False,
            "max_spring_dist": False,
            "floor_y": False,
            "floor_bounce": False,
        },
    )
    fitter._target_pyramids = _make_target_pyramids(
        state, cfg, vp, substeps=run_cfg.substeps, frames=4, num_levels=2
    )
    fitter._target_device = str(state.device)

    ok, msg = fitter.build_optimizer(state)
    assert not ok
    assert "trainable" in msg.lower()


def test_cfg_bounds_only_include_optimizable_simulation_params():
    assert set(DifferentiableFitter._CFG_BOUNDS) == {
        "time_step",
        "gravity_g",
        "spring_stiffness",
        "damping_stiffness",
        "floor_bounce",
    }


def test_step_reports_gradients_and_parameter_updates():
    cfg = SimulationConfig(
        time_step=0.01,
        gravity_g=0.0,
        floor_enabled=False,
        max_spring_dist=1.0,
    )
    run_cfg = RuntimeConfig(width=320, height=240, panel_width=80, substeps=3)
    vp = Viewport(
        -1.0, 1.0, -1.0, 1.0, 0, 0, run_cfg.width - run_cfg.panel_width, run_cfg.height
    )
    state = SimulationState.empty("cpu")
    state.add_particle((-0.2, 0.0), cfg, fixed=False)
    state.add_particle((0.2, 0.0), cfg, fixed=False)
    state.rest_len = state.rest_len * 0.6

    fitter = DifferentiableFitter(
        run_cfg, num_levels=3, sigma_world=0.05, kernel_size=5
    )
    fitter.configure(
        train_mass=True,
        train_edge_k=True,
        lr=0.01,
        min_advance_loss=0.001,
        temporal_discount=0.97,
        horizon_step=2,
        max_horizon=18,
    )

    target_state = state.clone()
    target_state.mass = target_state.mass * 1.35
    target_state.edge_k = target_state.edge_k * 0.72
    fitter._target_pyramids = _make_target_pyramids(
        target_state, cfg, vp, substeps=run_cfg.substeps, frames=6, num_levels=3
    )
    fitter._target_device = str(state.device)

    _build_optimizer(fitter, state)

    result = fitter.step(state, cfg, vp, run_cfg.substeps)
    assert result.ok
    assert result.mass_grad_norm > 0.0
    assert result.edge_k_grad_norm > 0.0
    assert (result.mass_param_delta > 0.0) or (result.edge_k_param_delta > 0.0)


def test_real_data_observable_and_video_optimize_one_step():
    repo_root = Path(__file__).resolve().parents[1]
    json_path = repo_root / "data" / "1" / "orig.json"
    video_path = repo_root / "data" / "1" / "orig.avi"
    if not json_path.exists() or not video_path.exists():
        return

    run_cfg = RuntimeConfig(width=1240, height=800, panel_width=420, substeps=4)
    cfg = SimulationConfig()
    state = load_observable_state(
        str(json_path),
        "cpu",
        default_spring_stiffness=cfg.spring_stiffness,
    )

    fitter = DifferentiableFitter(
        run_cfg,
        num_levels=run_cfg.render_pyramid_levels,
        sigma_world=run_cfg.render_particle_sigma_world,
        kernel_size=run_cfg.render_pyramid_kernel_size,
    )
    fitter.configure(
        train_mass=True,
        train_edge_k=True,
        lr=0.005,
        min_advance_loss=0.001,
        temporal_discount=0.97,
        horizon_step=2,
        max_horizon=96,
    )

    frame_count = fitter.load_target_video(
        str(video_path),
        device="cpu",
        scene_width=run_cfg.width - run_cfg.panel_width,
        scene_height=run_cfg.height,
    )
    assert frame_count > 0

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
    _build_optimizer(fitter, state)
    result = fitter.step(state, cfg, vp, run_cfg.substeps)
    assert result.ok
    assert result.loss >= 0.0


def test_mass_only_optimization_recovers_target_mass():
    cfg = SimulationConfig(
        time_step=0.01,
        gravity_g=0.0,
        floor_enabled=False,
        max_spring_dist=1.0,
    )
    run_cfg = RuntimeConfig(width=320, height=240, panel_width=80, substeps=3)
    vp = Viewport(
        -1.0, 1.0, -1.0, 1.0, 0, 0, run_cfg.width - run_cfg.panel_width, run_cfg.height
    )

    state = SimulationState.empty("cpu")
    state.add_particle((-0.2, 0.0), cfg, fixed=False)
    state.add_particle((0.2, 0.0), cfg, fixed=False)
    state.rest_len = state.rest_len * 0.6

    target_state = state.clone()
    target_state.mass = target_state.mass * 1.9

    fitter = DifferentiableFitter(
        run_cfg, num_levels=3, sigma_world=0.05, kernel_size=5
    )
    fitter.configure(
        train_mass=True,
        train_edge_k=False,
        lr=0.03,
        loss_threshold=0.0,
        temporal_discount=0.97,
        horizon_step=2,
        max_horizon=18,
        optimize_cfg={
            "time_step": False,
            "gravity_g": False,
            "particle_mass": False,
            "spring_stiffness": False,
            "damping_stiffness": False,
            "max_spring_dist": False,
            "floor_y": False,
            "floor_bounce": False,
        },
        auto_advance_no_improve=False,
    )

    fitter._target_pyramids = _make_target_pyramids(
        target_state, cfg, vp, substeps=run_cfg.substeps, frames=8, num_levels=3
    )
    fitter._target_device = str(state.device)

    _build_optimizer(fitter, state)

    before_mass_err = float(
        torch.mean(torch.abs(state.mass - target_state.mass)).item()
    )
    before_loss = fitter.step(state, cfg, vp, run_cfg.substeps).loss

    loss = before_loss
    for _ in range(12):
        loss = fitter.step(state, cfg, vp, run_cfg.substeps).loss

    after_mass_err = float(torch.mean(torch.abs(state.mass - target_state.mass)).item())
    assert after_mass_err < before_mass_err
    assert loss <= before_loss


def test_cfg_gravity_optimization_moves_toward_target():
    cfg = SimulationConfig(
        time_step=0.01,
        gravity_g=-2.0,
        floor_enabled=False,
        max_spring_dist=1.0,
    )
    run_cfg = RuntimeConfig(width=320, height=240, panel_width=80, substeps=3)
    vp = Viewport(
        -1.0, 1.0, -1.0, 1.0, 0, 0, run_cfg.width - run_cfg.panel_width, run_cfg.height
    )

    state = SimulationState.empty("cpu")
    state.add_particle((-0.15, 0.35), cfg, fixed=False)
    state.add_particle((0.15, 0.35), cfg, fixed=False)
    state.rest_len = state.rest_len * 0.8

    target_cfg = SimulationConfig(
        time_step=cfg.time_step,
        gravity_g=-10.0,
        floor_enabled=False,
        max_spring_dist=cfg.max_spring_dist,
        damping_stiffness=cfg.damping_stiffness,
        spring_stiffness=cfg.spring_stiffness,
    )

    fitter = DifferentiableFitter(
        run_cfg, num_levels=3, sigma_world=0.05, kernel_size=5
    )
    fitter.configure(
        train_mass=False,
        train_edge_k=False,
        lr=0.01,
        loss_threshold=0.0,
        temporal_discount=0.97,
        horizon_step=2,
        max_horizon=18,
        optimize_cfg={
            "time_step": False,
            "gravity_g": True,
            "particle_mass": False,
            "spring_stiffness": False,
            "damping_stiffness": False,
            "max_spring_dist": False,
            "floor_y": False,
            "floor_bounce": False,
        },
        cfg_step_scale=0.2,
        auto_advance_no_improve=False,
    )

    fitter._target_pyramids = _make_target_pyramids(
        state, target_cfg, vp, substeps=run_cfg.substeps, frames=8, num_levels=3
    )
    fitter._target_device = str(state.device)

    _build_optimizer(fitter, state)

    before_g_err = abs(cfg.gravity_g - target_cfg.gravity_g)
    before_loss = fitter.step(state, cfg, vp, run_cfg.substeps).loss

    loss = before_loss
    for _ in range(8):
        step = fitter.step(state, cfg, vp, run_cfg.substeps)
        loss = step.loss

    after_g_err = abs(cfg.gravity_g - target_cfg.gravity_g)
    assert after_g_err < before_g_err
    assert loss <= before_loss
