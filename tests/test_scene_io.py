import json
import random

import pytest

from pygoo.config import SimulationConfig
from pygoo.io.scene import (
    generate_random_state,
    load_observable_state,
    load_scene,
    randomize_config,
    save_scene,
)


def test_randomize_config_ranges():
    cfg = SimulationConfig()
    randomize_config(cfg, random.Random(123))

    assert 0.0001 <= cfg.time_step <= 0.001
    assert -18.0 <= cfg.gravity_g <= -0.5
    assert 0.08 <= cfg.particle_mass <= 12.0
    assert 25.0 <= cfg.spring_stiffness <= 850.0
    assert 0.0 <= cfg.damping_stiffness <= 12.0
    assert cfg.max_spring_dist == 0.25
    assert 0.05 <= cfg.floor_bounce <= 1.0


def test_generate_random_state_has_varied_particles_and_springs():
    cfg = SimulationConfig()
    randomize_config(cfg, random.Random(1))
    state = generate_random_state(cfg, "cpu", -2.0, 2.0, -1.0, 1.0, random.Random(2))

    assert state.pos.shape[0] >= 8
    assert state.pos.shape[0] <= 28
    assert state.mass.shape[0] == state.pos.shape[0]
    assert state.edges.shape[0] >= state.pos.shape[0] - 1
    assert len(set(float(x) for x in state.mass.tolist())) > 1
    assert len(set(float(x) for x in state.edge_k.tolist())) > 1


def test_generate_random_state_stays_within_view():
    cfg = SimulationConfig()
    rng = random.Random(9)
    x_min, x_max, y_min, y_max = -2.0, 2.0, -1.0, 1.0
    state = generate_random_state(cfg, "cpu", x_min, x_max, y_min, y_max, rng)

    x_margin = max((x_max - x_min) * 0.025, 0.08)
    y_margin = max((y_max - y_min) * 0.03, 0.06)
    x_lo = x_min + x_margin
    x_hi = x_max - x_margin
    y_lo = max(cfg.floor_y + y_margin, y_min + y_margin)
    y_hi = y_max - y_margin

    xs = state.pos[:, 0].tolist()
    ys = state.pos[:, 1].tolist()
    assert all(x_lo <= float(x) <= x_hi for x in xs)
    assert all(y_lo <= float(y) <= y_hi for y in ys)


def test_scene_round_trip(tmp_path):
    cfg = SimulationConfig()
    randomize_config(cfg, random.Random(4))
    state = generate_random_state(cfg, "cpu", -2.0, 2.0, -1.0, 1.0, random.Random(5))

    out = tmp_path / "scene.json"
    save_scene(str(out), cfg, state)
    cfg2, state2 = load_scene(str(out), "cpu")

    assert cfg2 == cfg
    assert state2.pos.shape == state.pos.shape
    assert state2.edges.shape == state.edges.shape
    assert state2.mass.tolist() == state.mass.tolist()


def test_load_observable_state_from_particles_and_springs(tmp_path):
    cfg = SimulationConfig(particle_mass=2.5, spring_stiffness=120.0)
    payload = {
        "particles": [
            {"x": 0.0, "y": 0.0, "mass": 2.0, "fixed": True},
            {"x": 0.3, "y": 0.0, "mass": 3.5},
            {"x": 0.3, "y": 0.4, "mass": 1.2, "fixed": False},
        ],
        "springs": [{"a": 0, "b": 1}, [1, 2]],
    }
    p = tmp_path / "observable.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    state = load_observable_state(str(p), "cpu", cfg)

    assert state.pos.shape == (3, 2)
    assert state.mass.tolist() == pytest.approx([1e6, 2.5, 2.5])
    assert state.fixed.tolist() == [True, False, False]
    assert state.edges.tolist() == [[0, 1], [1, 2]]
    assert state.rest_len.shape[0] == 2
    assert state.edge_k.shape[0] == 2
    assert state.edge_k.tolist() == pytest.approx([400.0, 300.0])


def test_load_observable_state_from_state_payload_includes_fixed(tmp_path):
    cfg = SimulationConfig(particle_mass=3.0, spring_stiffness=50.0)
    payload = {
        "state": {
            "pos": [[0.0, 0.0], [1.0, 0.0]],
            "mass": [1.0, 2.0],
            "fixed": [False, True],
            "edges": [[0, 1]],
        }
    }
    p = tmp_path / "observable_state_payload.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    state = load_observable_state(str(p), "cpu", cfg)

    assert state.fixed.tolist() == [False, True]
    assert state.mass.tolist() == pytest.approx([3.0, 1e6])
