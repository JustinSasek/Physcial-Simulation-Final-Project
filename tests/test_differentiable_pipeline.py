import pytest
import torch

from pygoo.config import SimulationConfig
from pygoo.physics.step import Stepper
from pygoo.render import Viewport, build_gaussian_pyramid, render_gaussian_pyramid
from pygoo.state import SimulationState


def _make_state() -> SimulationState:
    cfg = SimulationConfig(
        time_step=0.005,
        gravity_g=-9.8,
        damping_stiffness=0.2,
        floor_enabled=False,
    )
    state = SimulationState.empty("cpu")
    state.add_particle((-0.2, 0.3), cfg, fixed=False)
    state.add_particle((0.2, 0.1), cfg, fixed=False)
    state.add_particle((0.0, 0.35), cfg, fixed=False)
    if state.rest_len.numel() > 0:
        state.rest_len = state.rest_len * 0.6
    return state


def test_differentiable_multistep_pipeline_gradients():
    cfg = SimulationConfig(
        time_step=0.005,
        gravity_g=-9.8,
        damping_stiffness=0.2,
        floor_enabled=False,
    )
    state = _make_state()
    state.mass = state.mass.clone().detach().requires_grad_(True)
    state.edge_k = state.edge_k.clone().detach().requires_grad_(True)

    stepper = Stepper(cfg, use_compile=False)
    stepper.rollout(state, 32)

    vp = Viewport(-1.0, 1.0, -1.0, 1.0, 0, 0, 64, 64)
    levels = render_gaussian_pyramid(state, vp, 64, 64, num_levels=4, sigma_world=0.06)

    assert [tuple(level.shape) for level in levels] == [
        (64, 64, 3),
        (32, 32, 3),
        (16, 16, 3),
        (8, 8, 3),
    ]

    loss = torch.tensor(0.0)
    for level in levels:
        loss = (
            loss
            + (level[..., 0] * 0.7 + level[..., 1] * 0.2 + level[..., 2] * 0.1).mean()
        )
    loss.backward()

    assert state.mass.grad is not None
    assert state.edge_k.grad is not None
    assert torch.isfinite(state.mass.grad).all()
    assert torch.isfinite(state.edge_k.grad).all()
    assert state.mass.grad.abs().sum().item() > 0.0
    assert state.edge_k.grad.abs().sum().item() > 0.0


def test_compiled_rollout_is_differentiable():
    cfg = SimulationConfig(
        time_step=0.004,
        gravity_g=-8.0,
        damping_stiffness=0.1,
        floor_enabled=False,
    )
    state = _make_state()
    state.mass = state.mass.clone().detach().requires_grad_(True)
    state.edge_k = state.edge_k.clone().detach().requires_grad_(True)

    stepper = Stepper(cfg, use_compile=True)
    try:
        stepper.rollout(state, 12)
    except Exception as exc:
        pytest.skip(f"torch.compile unavailable in this environment: {exc}")

    vp = Viewport(-1.0, 1.0, -1.0, 1.0, 0, 0, 48, 48)
    level0 = render_gaussian_pyramid(state, vp, 48, 48, num_levels=1, sigma_world=0.06)[
        0
    ]
    loss = level0.mean()
    loss.backward()

    assert state.mass.grad is not None
    assert state.edge_k.grad is not None
    assert state.mass.grad.abs().sum().item() > 0.0
    assert state.edge_k.grad.abs().sum().item() > 0.0


def test_floor_enabled_pyramid_is_generated_from_level0():
    state = _make_state()
    vp = Viewport(-1.0, 1.0, -1.0, 1.0, 0, 0, 64, 64)

    levels = render_gaussian_pyramid(
        state,
        vp,
        64,
        64,
        num_levels=4,
        sigma_world=0.06,
        kernel_size=5,
        floor_y=-0.2,
        floor_enabled=True,
    )
    expected = build_gaussian_pyramid(levels[0], num_levels=4, kernel_size=5)

    for got, want in zip(levels, expected):
        assert torch.allclose(got, want, atol=1e-6, rtol=1e-5)
