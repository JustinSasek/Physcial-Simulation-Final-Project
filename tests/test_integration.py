import pytest

from pygoo.config import SimulationConfig
from pygoo.physics.step import Stepper
from pygoo.state import SimulationState


def test_step_gravity_moves_down():
    cfg = SimulationConfig(time_step=0.01, gravity_g=-9.8, springs_enabled=False)
    state = SimulationState.empty("cpu")
    state.add_particle((0.0, 0.2), cfg, fixed=False)
    y0 = state.pos[0, 1].item()

    stepper = Stepper(cfg, use_compile=False)
    stepper.step(state)
    v1 = state.vel[0, 1].item()
    y1 = state.pos[0, 1].item()
    stepper.step(state)

    assert v1 < 0.0
    assert y1 == y0
    assert state.pos[0, 1].item() < y0


def test_step_spring_reduces_stretch_with_damping():
    cfg = SimulationConfig(time_step=0.005, gravity_g=0.0, damping_stiffness=1.0)
    state = SimulationState.empty("cpu")
    state.add_particle((0.0, 0.0), cfg, fixed=False)
    state.add_particle((0.2, 0.0), cfg, fixed=False)
    state.rest_len = state.rest_len * 0.5

    d0 = abs((state.pos[1, 0] - state.pos[0, 0]).item())
    stepper = Stepper(cfg, use_compile=False)
    for _ in range(200):
        stepper.step(state)
    d1 = abs((state.pos[1, 0] - state.pos[0, 0]).item())

    assert d1 < d0


def _first_rebound_height(cfg: SimulationConfig, y0: float) -> float:
    state = SimulationState.empty("cpu")
    state.add_particle((0.0, y0), cfg, fixed=False)
    stepper = Stepper(cfg, use_compile=False)

    seen_bounce = False
    for _ in range(12000):
        vy_prev = state.vel[0, 1].item()
        stepper.step(state)
        if not seen_bounce and vy_prev < 0.0 and state.vel[0, 1].item() > 0.0:
            seen_bounce = True
        if seen_bounce and state.vel[0, 1].item() <= 0.0:
            return state.pos[0, 1].item() - cfg.floor_y
    raise AssertionError("did not observe rebound apex")


def test_floor_bounce_applies_restitution():
    cfg = SimulationConfig(
        time_step=0.01,
        gravity_g=-9.8,
        springs_enabled=False,
        damping_enabled=False,
        floor_enabled=True,
        floor_bounce=0.75,
    )
    state = SimulationState.empty("cpu")
    state.add_particle((0.0, cfg.floor_y + 0.005), cfg, fixed=False)
    state.vel[0, 1] = -1.0

    stepper = Stepper(cfg, use_compile=False)
    vy_impact = state.vel[0, 1].item() + cfg.time_step * cfg.gravity_g
    stepper.step(state)

    assert state.pos[0, 1].item() == cfg.floor_y
    assert state.vel[0, 1].item() > 0.0
    assert state.vel[0, 1].item() == pytest.approx(-cfg.floor_bounce * vy_impact)


def test_floor_bounce_height_scales_with_restitution():
    low_cfg = SimulationConfig(
        time_step=0.001,
        gravity_g=-9.8,
        springs_enabled=False,
        damping_enabled=False,
        floor_enabled=True,
        floor_bounce=0.35,
    )
    high_cfg = SimulationConfig(
        time_step=0.001,
        gravity_g=-9.8,
        springs_enabled=False,
        damping_enabled=False,
        floor_enabled=True,
        floor_bounce=0.9,
    )

    low_h = _first_rebound_height(low_cfg, y0=0.4)
    high_h = _first_rebound_height(high_cfg, y0=0.4)

    assert low_h > 0.0
    assert high_h > low_h * 3.0
